// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JPEGLI_TRELLIS_H_
#define LIB_JPEGLI_TRELLIS_H_

#include "lib/jpegli/common.h"

namespace jpegli {

// Applies trellis quantization to AC coefficients to reduce entropy coding cost.
// Returns true when trellis coding was applied.
bool ApplyTrellisQuantization(j_compress_ptr cinfo);

}  // namespace jpegli

#endif  // LIB_JPEGLI_TRELLIS_H_
