/* CUDA 3D neighborood histogram
 *
 * Originally written by Alessandro Mirone for the ASEMI project.
 * Integrated and updated by Johann A. Briffa
 */

#define MAXBINLIMS 41

__device__ __constant__ float bins_limits[MAXBINLIMS];

/* Dynamic shared memory region
 *
 * The first WW_X * WW_Y elements hold the tile neighbourhood, where:
 *    WW_X = (RADIUS_H + blockDim.x + RADIUS_H)
 *    WW_Y = (RADIUS_H + blockDim.y + RADIUS_H)
 *
 * The next blockDim.y * blockDim.x * NBINS elements hold the histograms for
 * each thread (i.e. a separate histogram for every voxel in the tile).
 */

extern __shared__ float SLICE[];

// Macro to determine absolute index for i'th bin of this thread's histogram

#define myhisto(i) SLICE[WW_Y * WW_X + (i) * blockDim.y * blockDim.x + tid]

/* Computes the contribution to the histogram from the slice at global_z.
 * (Auxiliary Function)
 *
 * Optimisations in this function include:
 *
 * 1) Parallel reading of tile neighbourhood into shared memory.
 *    Once this neighbourhood is loaded in shared memory, all the threads in
 *    the block can access the voxels of the current slice from shared memory.
 *
 * 2) Neighbourhood access by warp threads when writing into shared memory is
 *    on different memory banks because successive threads access adjacent
 *    shared memory locations.
 *
 * 3) Neighbourhood access by warp threads when reading from shared memory is
 *    also on different memory banks because again successive threads access
 *    adjacent shared memory locations, with threadIdx.x carying over a range
 *    of width 16. There could be a conflict when WW_X < 16 or when WW_X > 16
 *    but not a multiple of 16. (Need to check last statement.)
 *
 * 4) Histogram for each thread is also held in shared memory.
 *    This speeds up computation and facilitates holding intermediate values
 *    between slice updates.
 *
 * 5) Histogram elements for thread A and thread B are on different shared
 *    memory banks.
 */
__device__ void update_slice(
      // iadd = +1/-1 to add/subtract
      const int iadd,
      // thread index in CUDA block ( tid >= 0 && tid < blockDim.y * blockDim.x )
      const int tid,
      // size of the section of slice read in this block
      const int WW_X, const int WW_Y,
      // neighbourhood radius around voxel for computing histogram
      const int RADIUS_H,
      // x,y coordinates of block's corner in global volume
      const int block_cx, const int block_cy,
      // z coordinate of thread's voxel in global volume
      const int global_z,
      // number of bins in histogram
      const int NBINS,
      // input volume and size
      const float *d_volume_in,
      const int NX, const int NY, const int NZ)
   {
   // parallel read of section of slice into shared memory
   for (int i_in_tile = tid;
         i_in_tile < WW_Y * WW_X;
         i_in_tile += blockDim.y * blockDim.x)
      {
      // determine x,y coordinates for this thread
      const int global_x = block_cx + (i_in_tile % WW_X) - RADIUS_H;
      const int global_y = block_cy + (i_in_tile / WW_X) - RADIUS_H;
      // read value at x,y from global memory if exists, zero otherwise
      float val = 0;
      if( global_x >= 0 && global_x < NX &&
          global_y >= 0 && global_y < NY &&
          global_z >= 0 && global_z < NZ )
         val = d_volume_in[global_x + NX * (global_y + NY * global_z)];
      // convert value to corresponding histogram bin index
      float res = -1;
      for(int i = 0; i < NBINS + 1; i++)
         if(val >= bins_limits[i])
            res = i;
      // store in shared memory
      SLICE[i_in_tile] = res;
      }
   // make sure all slice section is read
   __syncthreads();
   // each thread computes histogram for neighbourhood of its coordinate
   for (int sx = -RADIUS_H; sx <= RADIUS_H; sx++)
      for (int sy = -RADIUS_H; sy <= RADIUS_H; sy++)
         {
         // get neighbouring pixel value (bin index, really)
         const float v = SLICE[threadIdx.x + sx + RADIUS_H + WW_X * (threadIdx.y + sy + RADIUS_H)];
         // add to histogram if within limits
         if (v >= 0 && v < NBINS)
            myhisto(int(v)) += iadd;
         }
   // make sure all slice section is processed
   __syncthreads();
   }

/* Computes the neighbourhood histogram for a given range of voxels.
 * (Main Kernel)
 *
 * Kernel is launched with a 2D grid of 2D blocks, covering a single slice of
 * voxels in the XY plane. A range of slices are considered with an internal
 * loop over the Z dimension. For each voxel (ix,iy,iz) we need to compute the
 * neighbourhood histogram.
 *
 * Storage is always in row-major order, so from slow to fast the dimensions
 * are Z, Y, X, or slice, row, column.
 *
 * The histograms of all voxels in a slice are computed in parallel, with
 * the computation for voxels in the same block benefiting from the use of
 * shared memory.
 *
 * The histograms for the first slice are computing by summing over the
 * contributions from slices making up the neighbourhood (±radius) in a loop.
 * Histograms for successive slices (iz) are computed using a rolling histogram,
 * that is:
 *    - subtracting the contribution from slice iz-radius-1, which does not
 *      contribute to the histograms of this slice (iz)
 *    - adding the contribution from slice iz+radius, which did not contribute
 *      to the histograms already computed
 */
__global__ void histogram_3d(
      // histogram output matrix
      float *d_volume_histo,
      // number of bins in histogram
      const int NBINS,
      // input volume and size
      const float *d_volume_in,
      const int NX, const int NY, const int NZ,
      // range of slices to consider in each dimension (half-open)
      const int x_start, const int x_stop,
      const int y_start, const int y_stop,
      const int z_start, const int z_stop,
      // il raggio della zone di interesse intorno al voxel
      const int RADIUS_H)
   {
   // size of the section of slice read in this block
   const int WW_X = RADIUS_H + blockDim.x + RADIUS_H;
   const int WW_Y = RADIUS_H + blockDim.y + RADIUS_H;
   // x,y coordinates of block's corner in global volume
   const int block_cx = x_start + blockIdx.x * blockDim.x;
   const int block_cy = y_start + blockIdx.y * blockDim.y;
   // thread index in CUDA block ( tid >= 0 && tid < blockDim.y * blockDim.x )
   const int tid = threadIdx.y * blockDim.x + threadIdx.x;
   // size of the histogram output matrix
   const int HX = x_stop - x_start;
   const int HY = y_stop - y_start;
   const int HZ = z_stop - z_start;
   // x,y coordinates of thread's voxel in histogram output matrix
   const int ix = block_cx + threadIdx.x - x_start;
   const int iy = block_cy + threadIdx.y - y_start;
   // z coordinate of thread's voxel in histogram output matrix (loop variable)
   int iz = 0;

   // every thread clears its histogram in shared memory
   for (int i = 0; i < NBINS; i++)
      myhisto(i) = 0;
   // make sure all threads have reset their histogram
   __syncthreads();

   // computation of first slice
   for (int z_offset = -RADIUS_H; z_offset <= RADIUS_H; z_offset++)
      {
      update_slice(+1, tid, WW_X, WW_Y, RADIUS_H, block_cx, block_cy,
            z_start + iz + z_offset,
            NBINS, d_volume_in, NX, NY, NZ);
      }
   // copy histogram from shared memory to global memory (non-coalesced)
   for (int ibin = 0; ibin < NBINS; ibin++)
      {
      if (ix < HX && iy < HY)
         d_volume_histo[ibin + NBINS * (ix + HX * (iy + HY * iz))] =
               myhisto(ibin);
      }
   // computation of following slices
   for (iz++; iz < HZ; iz++)
      {
      update_slice(-1, tid, WW_X, WW_Y, RADIUS_H, block_cx, block_cy,
            z_start + iz - 1 - RADIUS_H,
            NBINS, d_volume_in, NX, NY, NZ);
      update_slice(+1, tid, WW_X, WW_Y, RADIUS_H, block_cx, block_cy,
            z_start + iz + RADIUS_H,
            NBINS, d_volume_in, NX, NY, NZ);
      // copy histogram from shared memory to global memory (non-coalesced)
      for (int ibin = 0; ibin < NBINS; ibin++)
         {
         if (ix < HX && iy < HY)
            d_volume_histo[ibin + NBINS * (ix + HX * (iy + HY * iz))] =
                  myhisto(ibin);
         }
      }
   }
