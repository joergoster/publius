// Publius - Didactic public domain bitboard chess engine 
// by Pawel Koziol

// NNUE evaluation. Net architecture and constants make it
// equivalent to the simple example provided by the bullet trainer:
// https://github.com/jw1912/bullet/blob/main/examples/simple.rs
// The architecture is 768 -> (N from 16 to 256)x2 -> 1.
// Publius is able to use neworks with hidden neuron count
// from 16 to 256, as long as it is a multiple of 16.

// This code draws some inspiration from Iris chess engine
// (https://github.com/citrus610/iris):
// I used it to check the math, separating NNUEparameters
// struct helped to create a nice file loader and the convention
// of using this-> helps to notice class members.

// On top of that, there is an optional AVX2 fast path. It works
// the  same  as the simple addition of accumulator  values, but 
// processes 16 int16 lanes at a time using 256-bit vectors. 
//
// If AVX2 isn't available, we can still compile the scalar code 
// (#ifndef part), and performance is still correct - just slower.

#include "types.h"
#include "piece.h"
#include "nn.h"
#include "publius.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

    // Constructor
    Net::Net() {
        this->Clear();
    }

    // Helper function to read i16 value from the NNUE file
    static bool ReadI16(std::FILE* f, i16* dst, size_t count) {
        return std::fread(dst, sizeof(i16), count, f) == count;
    }

    // Load a network from the bullet-generated file. This loader
    // is nice because it can comfortably read and set up nets
    // with the hidden layer size between 16 and 256, as long as
    // that size is a multiple of 16.
    bool Net::LoadFromFile(const char* path) {

        std::FILE* f = std::fopen(path, "rb");
        if (!f) return false;

        // Measure file size
        std::fseek(f, 0, SEEK_END);
        long fileBytesL = std::ftell(f);
        std::rewind(f);

        if (fileBytesL <= 0) { 
            std::fclose(f); 
            return false; 
        }

        const size_t fileBytes = (size_t)fileBytesL;

        // Packed layout size in bytes (without tail padding):
        // bytes = (771*width + 1) * sizeof(i16) = 1542*width + 2
        auto packedBytes = [](size_t width) -> size_t {
            return 1542u * width + 2u;
            };

        // Pick N (any multiple of 16, 16..256) with smallest extra bytes
        constexpr size_t padding = 64; // allowed trailing padding/noise
        networkWidth = HIDDEN_SIZE;
        size_t bestExtra = (size_t)-1;

        for (size_t width = 16; width <= HIDDEN_SIZE; width += 16) {
            size_t need = packedBytes(width);
            if (fileBytes < need) continue;

            size_t extra = fileBytes - need;
            if (extra <= padding && extra < bestExtra) {
                bestExtra = extra;
                networkWidth = width;
            }
        }

        // Now that we know the network width, we can read it
        const size_t width = networkWidth;

        // Zero-fill so unused neurons [N..255] are inert
        std::memset(&PARAMS, 0, sizeof(PARAMS));

        // Read packed params into the first N columns
        for (size_t in = 0; in < INPUT_SIZE; ++in) {
            if (!ReadI16(f, &PARAMS.inputWeights[in][0], width)) {
                std::fclose(f);
                return false;
            }
        }

        // Read input biases
        if (!ReadI16(f, &PARAMS.inputBiases[0], width)) {
            std::fclose(f);
            return false;
        }

        // Read output weights
        if (!ReadI16(f, &PARAMS.outputWeights[0][0], width) ||
            !ReadI16(f, &PARAMS.outputWeights[1][0], width)) {
            std::fclose(f);
            return false;
        }

        // Read output bias
        if (std::fread(&PARAMS.outputBias, sizeof(i16), 1, f) != 1) {
            std::fclose(f);
            return false;
        }

        // Ignore whatever remains (tail padding)
        std::fclose(f);

        // Rebuild accumulator from loaded biases
        this->Clear();

        // Clear history and all the transposition tables
        OnNewGame();

        // Everything worked, the net has been read
        return true;
    }

    // Returns NNUE evaluation of position
    i32 Net::GetScore(i8 color) {

#if defined(__AVX2__)
        const i32 score = SumAccumulatorAVX2(color);
#else
        i32 score = 0;

        score += SumHalfAccumulator(
            this->accumulator[color],
            PARAMS.outputWeights[0]
        );

        score += SumHalfAccumulator(
            this->accumulator[!color],
            PARAMS.outputWeights[1]
        );
#endif

        return (score / L0_SCALE + PARAMS.outputBias)
            * EVAL_SCALE / MUL_SCALE;
    }

    // Sums the accumulated scoes for one side
    i32 Net::SumHalfAccumulator(i16 inputs[HIDDEN_SIZE], i16 weights[HIDDEN_SIZE]) {

        i32 value = 0;

        if (networkWidth < HIDDEN_SIZE)
            for (size_t i = 0; i < networkWidth; ++i) // works faster for smaller nets
                value += GetScrelu(inputs[i]) * weights[i];
        else
            for (size_t i = 0; i < HIDDEN_SIZE; ++i) // works faster for max network size (4% for hl = 256)
                value += GetScrelu(inputs[i]) * weights[i];

        return value;
    };

    // Sets indices for both network perspectives
    static inline void SetIndices(i8 color, i8 type, i8 sq, int& idxW, int& idxB) {
        idxW = Index(color, type, sq);
        idxB = Index(!color, type, sq ^ 56);
    }

    // Adds "a feature" (a piece on a square) to the accumulator
    void Net::Add(i8 color, i8 type, i8 square) {

        // We need two indices, for white and black part
        // of the accumulator
        const auto indexWhite = Index(color, type, square);
        const auto indexBlack = Index(!color, type, square^56);

#if defined(__AVX2__)
        AddAVX2(indexWhite, indexBlack);
#else
        // Update the accumulator
        if (networkWidth < HIDDEN_SIZE)
            for (size_t i = 0; i < networkWidth; ++i) { // works faster for smaller nets
                this->accumulator[0][i] += PARAMS.inputWeights[indexWhite][i];
                this->accumulator[1][i] += PARAMS.inputWeights[indexBlack][i];
            }
        else
            for (size_t i = 0; i < HIDDEN_SIZE; ++i) {  // works faster for max network size
                this->accumulator[0][i] += PARAMS.inputWeights[indexWhite][i];
                this->accumulator[1][i] += PARAMS.inputWeights[indexBlack][i];
            }
#endif
    }

    // Deletes a feature (a piece on a square) from the accumulator
    void Net::Del(i8 color, i8 type, i8 square) {

        const auto indexWhite = Index(color, type, square);
        const auto indexBlack = Index(!color, type, square^56);

#if defined(__AVX2__)
        DelAVX2(indexWhite, indexBlack);
#else
        if (networkWidth < HIDDEN_SIZE)
            for (size_t i = 0; i < networkWidth; ++i) { // works faster for smaller nets
                this->accumulator[0][i] -= PARAMS.inputWeights[indexWhite][i];
                this->accumulator[1][i] -= PARAMS.inputWeights[indexBlack][i];
            }
        else
            for (size_t i = 0; i < HIDDEN_SIZE; ++i) {  // works faster for max network size
                this->accumulator[0][i] -= PARAMS.inputWeights[indexWhite][i];
                this->accumulator[1][i] -= PARAMS.inputWeights[indexBlack][i];
            }
#endif
    }

    // a move operation performed on from and to squares at once
    // is slightly faster in AVX2 mode
    void Net::Move(i8 color, i8 type, i8 addSq, i8 subSq)
    {
        int addW, addB, subW, subB;
        SetIndices(color, type, addSq, addW, addB);
        SetIndices(color, type, subSq, subW, subB);

#if defined(__AVX2__)
        MoveAVX2(addW, addB, subW, subB);
#else
        if (networkWidth < HIDDEN_SIZE)
            for (size_t i = 0; i < networkWidth; ++i) {
                this->accumulator[0][i] += PARAMS.inputWeights[addW][i];
                this->accumulator[1][i] += PARAMS.inputWeights[addB][i];
                this->accumulator[0][i] -= PARAMS.inputWeights[subW][i];
                this->accumulator[1][i] -= PARAMS.inputWeights[subB][i];
            }
        else
            for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
                this->accumulator[0][i] += PARAMS.inputWeights[addW][i];
                this->accumulator[1][i] += PARAMS.inputWeights[addB][i];
                this->accumulator[0][i] -= PARAMS.inputWeights[subW][i];
                this->accumulator[1][i] -= PARAMS.inputWeights[subB][i];
            }
#endif
    }

    // Clears the net (sets the empty board state)
    void Net::Clear() {

        for (i8 color = 0; color < 2; ++color) {
            for (size_t i = 0; i < HIDDEN_SIZE; ++i) {
                this->accumulator[color][i] = PARAMS.inputBiases[i];
            }
        }
    }

    // "Cold start" - setting values for a new board position
    void Net::Refresh(Position& pos) {

        this->Clear();

        for (i8 sq = 0; sq < 64; ++sq) {
            const i8 piece = pos.GetPiece((Square)sq);

            if (piece == noPiece)
                continue;

            const i8 type = (i8)TypeOfPiece((ColoredPiece)piece);
            const i8 color = (i8)ColorOfPiece((ColoredPiece)piece);

            this->Add(color, type, sq);
        }
    }

#ifdef __AVX2__
    // networkWidth is always a multiple of 16, so AVX2 loops need no scalar tail.

    void Net::AddAVX2(int indexWhite, int indexBlack) {

        i16* __restrict a0 = &this->accumulator[0][0];
        i16* __restrict a1 = &this->accumulator[1][0];
        const i16* __restrict w0 = &PARAMS.inputWeights[indexWhite][0];
        const i16* __restrict w1 = &PARAMS.inputWeights[indexBlack][0];

        // 16 int16 lanes per __m256i
        for (size_t i = 0; i < networkWidth; i += 16) {
            __m256i A0 = _mm256_loadu_si256((const __m256i*)(a0 + i));
            __m256i W0 = _mm256_loadu_si256((const __m256i*)(w0 + i));
            __m256i A1 = _mm256_loadu_si256((const __m256i*)(a1 + i));
            __m256i W1 = _mm256_loadu_si256((const __m256i*)(w1 + i));

            A0 = _mm256_add_epi16(A0, W0);
            A1 = _mm256_add_epi16(A1, W1);

            _mm256_storeu_si256((__m256i*)(a0 + i), A0);
            _mm256_storeu_si256((__m256i*)(a1 + i), A1);
        }
    }

    void Net::DelAVX2(int indexWhite, int indexBlack) {

        i16* __restrict a0 = &this->accumulator[0][0];
        i16* __restrict a1 = &this->accumulator[1][0];
        const i16* __restrict w0 = &PARAMS.inputWeights[indexWhite][0];
        const i16* __restrict w1 = &PARAMS.inputWeights[indexBlack][0];

        for (size_t i = 0; i < networkWidth; i += 16) {
            __m256i A0 = _mm256_loadu_si256((const __m256i*)(a0 + i));
            __m256i W0 = _mm256_loadu_si256((const __m256i*)(w0 + i));
            __m256i A1 = _mm256_loadu_si256((const __m256i*)(a1 + i));
            __m256i W1 = _mm256_loadu_si256((const __m256i*)(w1 + i));

            A0 = _mm256_sub_epi16(A0, W0);
            A1 = _mm256_sub_epi16(A1, W1);

            _mm256_storeu_si256((__m256i*)(a0 + i), A0);
            _mm256_storeu_si256((__m256i*)(a1 + i), A1);
        }
    }

    void Net::MoveAVX2(int addW, int addB, int subW, int subB) {

        i16* __restrict a0 = &this->accumulator[0][0];
        i16* __restrict a1 = &this->accumulator[1][0];
        const i16* __restrict wAdd0 = &PARAMS.inputWeights[addW][0];
        const i16* __restrict wAdd1 = &PARAMS.inputWeights[addB][0];
        const i16* __restrict wSub0 = &PARAMS.inputWeights[subW][0];
        const i16* __restrict wSub1 = &PARAMS.inputWeights[subB][0];

        for (size_t i = 0; i < networkWidth; i += 16) {
            __m256i A0 = _mm256_loadu_si256((const __m256i*)(a0 + i));
            __m256i A1 = _mm256_loadu_si256((const __m256i*)(a1 + i));
            __m256i ADD0 = _mm256_loadu_si256((const __m256i*)(wAdd0 + i));
            __m256i ADD1 = _mm256_loadu_si256((const __m256i*)(wAdd1 + i));
            __m256i SUB0 = _mm256_loadu_si256((const __m256i*)(wSub0 + i));
            __m256i SUB1 = _mm256_loadu_si256((const __m256i*)(wSub1 + i));

            A0 = _mm256_add_epi16(A0, ADD0);
            A1 = _mm256_add_epi16(A1, ADD1);
            A0 = _mm256_sub_epi16(A0, SUB0);
            A1 = _mm256_sub_epi16(A1, SUB1);

            _mm256_storeu_si256((__m256i*)(a0 + i), A0);
            _mm256_storeu_si256((__m256i*)(a1 + i), A1);
        }
    }

    static inline i32 HorizontalSum256(__m256i v) {

        // Add upper 128 bits to lower 128 bits
        __m128i sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256(v, 1)
        );

        // Four i32 values -> one i32 value
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);

        return _mm_cvtsi128_si32(sum128);
    }


    // Calculates both accumulator perspectives in one pass.
    i32 Net::SumAccumulatorAVX2(i8 color) {

        const i16* __restrict inputs0 =
            &this->accumulator[color][0];

        const i16* __restrict inputs1 =
            &this->accumulator[!color][0];

        const i16* __restrict weights0 =
            &PARAMS.outputWeights[0][0];

        const i16* __restrict weights1 =
            &PARAMS.outputWeights[1][0];

        const __m256i zero = _mm256_setzero_si256();
        const __m256i upper = _mm256_set1_epi16((i16)L0_SCALE);

        // Separate accumulators reduce the dependency chain.
        __m256i sum0Lo = _mm256_setzero_si256();
        __m256i sum0Hi = _mm256_setzero_si256();
        __m256i sum1Lo = _mm256_setzero_si256();
        __m256i sum1Hi = _mm256_setzero_si256();

        for (size_t i = 0; i < networkWidth; i += 16) {

            // Load and clamp 16 int16 accumulator values to [0, 255].
            __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs0 + i));
            __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs1 + i));

            x0 = _mm256_max_epi16(x0, zero);
            x0 = _mm256_min_epi16(x0, upper);
            x1 = _mm256_max_epi16(x1, zero);
            x1 = _mm256_min_epi16(x1, upper);

            // Output weights remain signed int16.
            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights0 + i));
            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights1 + i));

            // Expand low and high groups of eight int16 values to i32.
            const __m128i x0Low128 = _mm256_castsi256_si128(x0);
            const __m128i x0High128 = _mm256_extracti128_si256(x0, 1);

            const __m128i x1Low128 = _mm256_castsi256_si128(x1);
            const __m128i x1High128 = _mm256_extracti128_si256(x1, 1);

            const __m128i w0Low128 = _mm256_castsi256_si128(w0);
            const __m128i w0High128 = _mm256_extracti128_si256(w0, 1);

            const __m128i w1Low128 = _mm256_castsi256_si128(w1);
            const __m128i w1High128 = _mm256_extracti128_si256(w1, 1);

            __m256i x0Lo = _mm256_cvtepi16_epi32(x0Low128);
            __m256i x0Hi = _mm256_cvtepi16_epi32(x0High128);

            __m256i x1Lo = _mm256_cvtepi16_epi32(x1Low128);
            __m256i x1Hi = _mm256_cvtepi16_epi32(x1High128);

            const __m256i w0Lo = _mm256_cvtepi16_epi32(w0Low128);
            const __m256i w0Hi = _mm256_cvtepi16_epi32(w0High128);

            const __m256i w1Lo = _mm256_cvtepi16_epi32(w1Low128);
            const __m256i w1Hi = _mm256_cvtepi16_epi32(w1High128);

            // SCReLU: x^2, followed by multiplication by output weight.
            x0Lo = _mm256_mullo_epi32(x0Lo, x0Lo);
            x0Hi = _mm256_mullo_epi32(x0Hi, x0Hi);
            x1Lo = _mm256_mullo_epi32(x1Lo, x1Lo);
            x1Hi = _mm256_mullo_epi32(x1Hi, x1Hi);

            sum0Lo = _mm256_add_epi32(sum0Lo, _mm256_mullo_epi32(x0Lo, w0Lo));
            sum0Hi = _mm256_add_epi32(sum0Hi, _mm256_mullo_epi32(x0Hi, w0Hi));
            sum1Lo = _mm256_add_epi32(sum1Lo, _mm256_mullo_epi32(x1Lo, w1Lo));
            sum1Hi = _mm256_add_epi32(sum1Hi, _mm256_mullo_epi32(x1Hi, w1Hi));
        }

        const __m256i total0 = _mm256_add_epi32(sum0Lo, sum0Hi);
        const __m256i total1 = _mm256_add_epi32(sum1Lo, sum1Hi);

        return HorizontalSum256(total0) + HorizontalSum256(total1);
    }

#endif