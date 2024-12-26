/***************************************************************************
 *  Copyright (C) 2018-2023 Andes Technology Corporation                   *
 *  All rights reserved.                                                   *
 ***************************************************************************/
#include "nds_type.h"
#ifdef ENA_VEC_INTRINSIC
#include <riscv_vector.h>
#endif

void ndsv_mat_mul_f32_c(float32_t * src1, float32_t * src2, float32_t * dst, uint32_t row, uint32_t col, uint32_t col2)
{
    const float32_t *A = src1;
    const float32_t *B = src2;
    const float32_t *InA = A;
    float32_t *C;
    float32_t sum;
    //double sum;
    uint32_t i, colcnt, col2cnt, rowcnt;

    i = 0u;
    rowcnt = row;
    do
    {
        C = dst + i;
        B = src2;

        /* Dot product of each row in src1 with each column in src2 */
        col2cnt = col2;
        do
        {
            sum = 0.0f;
            A = InA;

            /* column loop */
            colcnt = col;
            do
            {
                sum += (*A++) * *B;
                B += col2;
                colcnt--;
            }
            while (colcnt != 0u);
            *C++ = sum;
            //*C++ = (float32_t)sum;
            col2cnt--;
            B = src2 + (col2 - col2cnt);
        }
        while (col2cnt != 0u);
        i = i + col2;
        InA = A;
        rowcnt--;
    }
    while (rowcnt != 0u);
}

void ndsv_mat_mul_f32_v(float32_t * src1, float32_t * src2, float32_t * dst, uint32_t row, uint32_t col, uint32_t col2)
{
	const uint32_t tiling_size = 128;
	int row_4 = row >> 2;
	int max_row = row_4 << 2;

	const float32_t* A = src1;
	const float32_t* B = src2;
	float32_t* C = dst;
	int r, k, kk, cc, vl;
	int kk_tiling_size, cc_tiling_size, max_kk_tiling_size;
	const float32_t* A11, * B11, \
				   * A21, * B21, \
				   * A31, * B31, \
				   * A41, * B41;
	float32_t * C1, * C2, * C3, * C4;
#ifdef ENA_VEC_INTRINSIC
	for (kk = 0; kk < col; kk += tiling_size)
	{
		kk_tiling_size = ((kk + tiling_size) > col ? col : (kk + tiling_size));
		if ((kk + tiling_size) <= col)
		{
			max_kk_tiling_size = kk_tiling_size - (tiling_size & 3);
		}
		else
		{
			max_kk_tiling_size = col - ((col - kk) & 3);
		}
		for (cc = 0; cc < col2; cc += tiling_size)
		{
			cc_tiling_size = ((cc + tiling_size) > col2 ? col2 : (cc + tiling_size));
			for (r = 0; r < max_row; r += 4)
			{
				int mulLen = cc_tiling_size - cc;
				while (mulLen > 0)
				{
					vl = __riscv_vsetvl_e32m4(mulLen);
					vfloat32m4_t v_reg0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
					vfloat32m4_t v_reg4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
					vfloat32m4_t v_reg8 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
					vfloat32m4_t v_reg12 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

					A11 = A + r * col + kk;
					A21 = A11 + col;
					A31 = A21 + col;
					A41 = A31 + col;
					B11 = B + kk * col2 + cc_tiling_size - mulLen;
					B21 = B11 + col2;
					B31 = B21 + col2;
					B41 = B31 + col2;
					C1 = C + r * col2 + cc_tiling_size - mulLen;
					C2 = C1 + col2;
					C3 = C2 + col2;
					C4 = C3 + col2;
					for (k = kk; k < max_kk_tiling_size; k += 4)
					{
						vfloat32m4_t v_reg16 = __riscv_vle32_v_f32m4(B11, vl);
						vfloat32m4_t v_reg20 = __riscv_vle32_v_f32m4(B21, vl);
						vfloat32m4_t v_reg24 = __riscv_vle32_v_f32m4(B31, vl);
						vfloat32m4_t v_reg28 = __riscv_vle32_v_f32m4(B41, vl);

						float32_t tmpa11 = *A11++;
						float32_t tmpa21 = *A21++;
						float32_t tmpa31 = *A31++;
						float32_t tmpa41 = *A41++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa11, v_reg16, vl);
						v_reg4 = __riscv_vfmacc(v_reg4, tmpa21, v_reg16, vl);
						v_reg8 = __riscv_vfmacc(v_reg8, tmpa31, v_reg16, vl);
						v_reg12 = __riscv_vfmacc(v_reg12, tmpa41, v_reg16, vl);

						float32_t tmpa12 = *A11++;
						float32_t tmpa22 = *A21++;
						float32_t tmpa32 = *A31++;
						float32_t tmpa42 = *A41++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa12, v_reg20, vl);
						v_reg4 = __riscv_vfmacc(v_reg4, tmpa22, v_reg20, vl);
						v_reg8 = __riscv_vfmacc(v_reg8, tmpa32, v_reg20, vl);
						v_reg12 = __riscv_vfmacc(v_reg12, tmpa42, v_reg20, vl);

						float32_t tmpa13 = *A11++;
						float32_t tmpa23 = *A21++;
						float32_t tmpa33 = *A31++;
						float32_t tmpa43 = *A41++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa13, v_reg24, vl);
						v_reg4 = __riscv_vfmacc(v_reg4, tmpa23, v_reg24, vl);
						v_reg8 = __riscv_vfmacc(v_reg8, tmpa33, v_reg24, vl);
						v_reg12 = __riscv_vfmacc(v_reg12, tmpa43, v_reg24, vl);

						float32_t tmpa14 = *A11++;
						float32_t tmpa24 = *A21++;
						float32_t tmpa34 = *A31++;
						float32_t tmpa44 = *A41++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa14, v_reg28, vl);
						v_reg4 = __riscv_vfmacc(v_reg4, tmpa24, v_reg28, vl);
						v_reg8 = __riscv_vfmacc(v_reg8, tmpa34, v_reg28, vl);
						v_reg12 = __riscv_vfmacc(v_reg12, tmpa44, v_reg28, vl);

						B11 += 4 * col2;
						B21 += 4 * col2;
						B31 += 4 * col2;
						B41 += 4 * col2;
					}

					for (; k < kk_tiling_size; k++)
					{
						vfloat32m4_t v_reg16 = __riscv_vle32_v_f32m4(B11, vl);

						float32_t tmpa11 = *A11++;
						float32_t tmpa21 = *A21++;
						float32_t tmpa31 = *A31++;
						float32_t tmpa41 = *A41++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa11, v_reg16, vl);
						v_reg4 = __riscv_vfmacc(v_reg4, tmpa21, v_reg16, vl);
						v_reg8 = __riscv_vfmacc(v_reg8, tmpa31, v_reg16, vl);
						v_reg12 = __riscv_vfmacc(v_reg12, tmpa41, v_reg16, vl);
						B11 += col2;
					}
					vfloat32m4_t v_reg16 = __riscv_vle32_v_f32m4(C1, vl);
					vfloat32m4_t v_reg20 = __riscv_vle32_v_f32m4(C2, vl);
					vfloat32m4_t v_reg24 = __riscv_vle32_v_f32m4(C3, vl);
					vfloat32m4_t v_reg28 = __riscv_vle32_v_f32m4(C4, vl);
					v_reg0 = __riscv_vfadd(v_reg0, v_reg16, vl);
					v_reg4 = __riscv_vfadd(v_reg4, v_reg20, vl);
					v_reg8 = __riscv_vfadd(v_reg8, v_reg24, vl);
					v_reg12 = __riscv_vfadd(v_reg12, v_reg28, vl);

					__riscv_vse32(C1, v_reg0, vl);
					__riscv_vse32(C2, v_reg4, vl);
					__riscv_vse32(C3, v_reg8, vl);
					__riscv_vse32(C4, v_reg12, vl);
					mulLen -= vl;
				}
			}
			for (; r < row; r++)
			{
				int mulLen = cc_tiling_size - cc;
				while (mulLen > 0)
				{
					vl = __riscv_vsetvl_e32m4(mulLen);
					vfloat32m4_t v_reg0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

					A11 = A + r * col + kk;
					B11 = B + kk * col2 + cc_tiling_size - mulLen;
					B21 = B11 + col2;
					B31 = B21 + col2;
					B41 = B31 + col2;
					C1 = C + r * col2 + cc_tiling_size - mulLen;
					for (k = kk; k < max_kk_tiling_size; k += 4)
					{
						vfloat32m4_t v_reg16 = __riscv_vle32_v_f32m4(B11, vl);
						vfloat32m4_t v_reg20 = __riscv_vle32_v_f32m4(B21, vl);
						vfloat32m4_t v_reg24 = __riscv_vle32_v_f32m4(B31, vl);
						vfloat32m4_t v_reg28 = __riscv_vle32_v_f32m4(B41, vl);

						float32_t tmpa11 = *A11++;
						float32_t tmpa12 = *A11++;
						float32_t tmpa13 = *A11++;
						float32_t tmpa14 = *A11++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa11, v_reg16, vl);
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa12, v_reg20, vl);
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa13, v_reg24, vl);
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa14, v_reg28, vl);
						B11 += 4 * col2;
						B21 += 4 * col2;
						B31 += 4 * col2;
						B41 += 4 * col2;
					}
					for (; k < kk_tiling_size; k++)
					{
						vfloat32m4_t v_reg20 = __riscv_vle32_v_f32m4(B11, vl);
						float32_t tmpa11 = *A11++;
						v_reg0 = __riscv_vfmacc(v_reg0, tmpa11, v_reg20, vl);
						B11 += col2;
					}
					vfloat32m4_t v_reg16 = __riscv_vle32_v_f32m4(C1, vl);
					v_reg0 = __riscv_vfadd(v_reg0, v_reg16, vl);
					__riscv_vse32(C1, v_reg0, vl);
					mulLen -= vl;
				}
			}
		}
	}
#else
	for (kk = 0; kk < col; kk += tiling_size)
	{
		kk_tiling_size = ((kk + tiling_size) > col ? col : (kk + tiling_size));
		if ((kk + tiling_size) <= col)
		{
			max_kk_tiling_size = kk_tiling_size - (tiling_size & 3);
		}
		else
		{
			max_kk_tiling_size = col - ((col - kk) & 3);
		}
		for (cc = 0; cc < col2; cc += tiling_size)
		{
			cc_tiling_size = ((cc + tiling_size) > col2 ? col2 : (cc + tiling_size));
			for (r = 0; r < max_row; r += 4)
			{
				int mulLen = cc_tiling_size - cc;
				while (mulLen > 0)
				{
					__asm__ __volatile__ ("vsetvli %[out], %[avl], " "e32" ", " "m4" ", " "tu" ", " "mu" "\n" : [out] "=r" (vl) : [avl] "r" (mulLen));
					__asm__ __volatile__("vand.vi" " " "v0" ", " "v0" ", %[imm]\n" :: [imm] "i" (0));
					__asm__ __volatile__("vand.vi" " " "v4" ", " "v4" ", %[imm]\n" :: [imm] "i" (0));
					__asm__ __volatile__("vand.vi" " " "v8" ", " "v8" ", %[imm]\n" :: [imm] "i" (0));
					__asm__ __volatile__("vand.vi" " " "v12" ", " "v12" ", %[imm]\n" :: [imm] "i" (0));

					A11 = A + r * col + kk;
					A21 = A11 + col;
					A31 = A21 + col;
					A41 = A31 + col;
					B11 = B + kk * col2 + cc_tiling_size - mulLen;
					B21 = B11 + col2;
					B31 = B21 + col2;
					B41 = B31 + col2;
					C1 = C + r * col2 + cc_tiling_size - mulLen;
					C2 = C1 + col2;
					C3 = C2 + col2;
					C4 = C3 + col2;
					for (k = kk; k < max_kk_tiling_size; k += 4)
					{
						__asm__ __volatile__("vle32.v" " " "v16" ", %[ptr]\n" :: [ptr] "A" (*(B11)));
						__asm__ __volatile__("vle32.v" " " "v20" ", %[ptr]\n" :: [ptr] "A" (*(B21)));
						__asm__ __volatile__("vle32.v" " " "v24" ", %[ptr]\n" :: [ptr] "A" (*(B31)));
						__asm__ __volatile__("vle32.v" " " "v28" ", %[ptr]\n" :: [ptr] "A" (*(B41)));
						float32_t tmpa11 = *A11++;
						float32_t tmpa21 = *A21++;
						float32_t tmpa31 = *A31++;
						float32_t tmpa41 = *A41++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa11));
						__asm__ __volatile__("vfmacc.vf" " " "v4" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa21));
						__asm__ __volatile__("vfmacc.vf" " " "v8" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa31));
						__asm__ __volatile__("vfmacc.vf" " " "v12" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa41));

						float32_t tmpa12 = *A11++;
						float32_t tmpa22 = *A21++;
						float32_t tmpa32 = *A31++;
						float32_t tmpa42 = *A41++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa12));
						__asm__ __volatile__("vfmacc.vf" " " "v4" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa22));
						__asm__ __volatile__("vfmacc.vf" " " "v8" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa32));
						__asm__ __volatile__("vfmacc.vf" " " "v12" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa42));

						float32_t tmpa13 = *A11++;
						float32_t tmpa23 = *A21++;
						float32_t tmpa33 = *A31++;
						float32_t tmpa43 = *A41++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v24" "\n" :: [rs1] "f" (tmpa13));
						__asm__ __volatile__("vfmacc.vf" " " "v4" ", %[rs1], " "v24" "\n" :: [rs1] "f" (tmpa23));
						__asm__ __volatile__("vfmacc.vf" " " "v8" ", %[rs1], " "v24" "\n" :: [rs1] "f" (tmpa33));
						__asm__ __volatile__("vfmacc.vf" " " "v12" ", %[rs1], " "v24" "\n" :: [rs1] "f" (tmpa43));

						float32_t tmpa14 = *A11++;
						float32_t tmpa24 = *A21++;
						float32_t tmpa34 = *A31++;
						float32_t tmpa44 = *A41++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v28" "\n" :: [rs1] "f" (tmpa14));
						__asm__ __volatile__("vfmacc.vf" " " "v4" ", %[rs1], " "v28" "\n" :: [rs1] "f" (tmpa24));
						__asm__ __volatile__("vfmacc.vf" " " "v8" ", %[rs1], " "v28" "\n" :: [rs1] "f" (tmpa34));
						__asm__ __volatile__("vfmacc.vf" " " "v12" ", %[rs1], " "v28" "\n" :: [rs1] "f" (tmpa44));

						B11 += 4 * col2;
						B21 += 4 * col2;
						B31 += 4 * col2;
						B41 += 4 * col2;
					}
					for (; k < kk_tiling_size; k++)
					{
						__asm__ __volatile__("vle32.v" " " "v16" ", %[ptr]\n" :: [ptr] "A" (*(B11)));
						float32_t tmpa11 = *A11++;
						float32_t tmpa21 = *A21++;
						float32_t tmpa31 = *A31++;
						float32_t tmpa41 = *A41++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa11));
						__asm__ __volatile__("vfmacc.vf" " " "v4" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa21));
						__asm__ __volatile__("vfmacc.vf" " " "v8" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa31));
						__asm__ __volatile__("vfmacc.vf" " " "v12" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa41));

						B11 += col2;
					}
					__asm__ __volatile__("vle32.v" " " "v16" ", %[ptr]\n" :: [ptr] "A" (*(C1)));
					__asm__ __volatile__("vle32.v" " " "v20" ", %[ptr]\n" :: [ptr] "A" (*(C2)));
					__asm__ __volatile__("vle32.v" " " "v24" ", %[ptr]\n" :: [ptr] "A" (*(C3)));
					__asm__ __volatile__("vle32.v" " " "v28" ", %[ptr]\n" :: [ptr] "A" (*(C4)));

					__asm__ __volatile__("vfadd.vv" " " "v0" ", " "v0" ", " "v16" "\n");
					__asm__ __volatile__("vfadd.vv" " " "v4" ", " "v4" ", " "v20" "\n");
					__asm__ __volatile__("vfadd.vv" " " "v8" ", " "v8" ", " "v24" "\n");
					__asm__ __volatile__("vfadd.vv" " " "v12" ", " "v12" ", " "v28" "\n");

					__asm__ __volatile__("vse32.v" " " "v0" ", %[ptr]\n" :: [ptr] "A" (*(C1)));
					__asm__ __volatile__("vse32.v" " " "v4" ", %[ptr]\n" :: [ptr] "A" (*(C2)));
					__asm__ __volatile__("vse32.v" " " "v8" ", %[ptr]\n" :: [ptr] "A" (*(C3)));
					__asm__ __volatile__("vse32.v" " " "v12" ", %[ptr]\n" :: [ptr] "A" (*(C4)));
					mulLen -= vl;
				}
			}
			for (; r < row; r++)
			{
				int mulLen = cc_tiling_size - cc;
				while (mulLen > 0)
				{
					__asm__ __volatile__ ("vsetvli %[out], %[avl], " "e32" ", " "m4" ", " "tu" ", " "mu" "\n" : [out] "=r" (vl) : [avl] "r" (mulLen));
					__asm__ __volatile__("vand.vi" " " "v0" ", " "v0" ", %[imm]\n" :: [imm] "i" (0));
					A11 = A + r * col + kk;
					B11 = B + kk * col2 + cc_tiling_size - mulLen;
					B21 = B11 + col2;
					B31 = B21 + col2;
					B41 = B31 + col2;
					C1 = C + r * col2 + cc_tiling_size - mulLen;
					for (k = kk; k < max_kk_tiling_size; k += 4)
					{
						__asm__ __volatile__("vle32.v" " " "v16" ", %[ptr]\n" :: [ptr] "A" (*(B11)));
						__asm__ __volatile__("vle32.v" " " "v20" ", %[ptr]\n" :: [ptr] "A" (*(B21)));
						__asm__ __volatile__("vle32.v" " " "v24" ", %[ptr]\n" :: [ptr] "A" (*(B31)));
						__asm__ __volatile__("vle32.v" " " "v28" ", %[ptr]\n" :: [ptr] "A" (*(B41)));
						float32_t tmpa11 = *A11++;
						float32_t tmpa12 = *A11++;
						float32_t tmpa13 = *A11++;
						float32_t tmpa14 = *A11++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v16" "\n" :: [rs1] "f" (tmpa11));
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa12));
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v24" "\n" :: [rs1] "f" (tmpa13));
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v28" "\n" :: [rs1] "f" (tmpa14));

						B11 += 4 * col2;
						B21 += 4 * col2;
						B31 += 4 * col2;
						B41 += 4 * col2;
					}
					for (; k < kk_tiling_size; k++)
					{
						__asm__ __volatile__("vle32.v" " " "v20" ", %[ptr]\n" :: [ptr] "A" (*(B11)));
						float32_t tmpa11 = *A11++;
						__asm__ __volatile__("vfmacc.vf" " " "v0" ", %[rs1], " "v20" "\n" :: [rs1] "f" (tmpa11));
						B11 += col2;
					}
					__asm__ __volatile__("vle32.v" " " "v16" ", %[ptr]\n" :: [ptr] "A" (*(C1)));
					__asm__ __volatile__("vfadd.vv" " " "v0" ", " "v0" ", " "v16" "\n");
					__asm__ __volatile__("vse32.v" " " "v0" ", %[ptr]\n" :: [ptr] "A" (*(C1)));
					mulLen -= vl;
				}
			}
		}
	}
#endif
}
