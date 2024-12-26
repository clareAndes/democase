/***************************************************************************
 *  Copyright (C) 2018-2023 Andes Technology Corporation                   *
 *  All rights reserved.                                                   *
 ***************************************************************************/

#include "nds_type.h"

void ndsv_mat_mul_f32_c(float32_t * src1,
	                    float32_t * src2,
	                    float32_t * dst,
	                    uint32_t row,
	                    uint32_t col,
	                    uint32_t col2);

void ndsv_mat_mul_f32_v(float32_t * src1,
                      	float32_t * src2,
	                    float32_t * dst,
                	  	uint32_t row,
                      	uint32_t col,
                       	uint32_t col2);
