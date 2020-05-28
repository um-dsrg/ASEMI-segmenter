/* CUDA 3D neighborood histogram
 *
 * Originally written by Alessandro Mirone for the ASEMI project.
 * Integrated and updated by Johann A. Briffa
 */

#define MAXBINLIMS 41

__device__ __constant__ float bins_limits[MAXBINLIMS];

// ogni blocco legge un pezzo di slice nella memoria shared
//
// ogni thread tiene il suo histogramma nella memoria share
// si usa la variabile SLICE

extern __shared__ float SLICE[];

//  WW_Y e WW_X sono le dimensioni del pezzo di slice che si legge.
// Al di sopra si pone anche l'istogramma privatizzato ( i e' l'indice dell'istogramma, tid e' il numero della thread)

#define myhisto(i) SLICE[WW_Y * WW_X + (i) * blockDim.y * blockDim.x + tid]

// ========================================================================
// update_slice : FUNZIONE AUSILIARIA CHIAMATA DAL KERNEL PRINCIPALE
//

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

/*  =======================================
 * ISTOGRAMMA: Kernel principale
 *
 * Istogramma is launched with a 2D grid of 2D blocks.
 * The dimensions of each block are (from slow (Y) to fast (X) )  WS_Y, WS_X
 * A block, then, works on a tile of a 2D slice, and, inside the kernel,
 * there is a loop on the Z dimension, named iz.
 *
 * The block and grid coordinates are translated to (iy,ix) which are the
 * planar coordinates of a voxel. The z coordinate is represented by the loop
 * variable iz.
 *
 * Now, imagine that for a given voxel at (iz,iy,ix) you know the histogram.
 * Then the variable iz increase by one. How do we obtain the histogram for
 * voxel at (iz,iy,ix) when we know already the histogram at (iz-1,iy,ix)?
 * We remove the contribution from slice  iz-1 - RADIUS_H
 * the we add the contribution from slice iz + RADIUS_H
 *
 * This is done by calling
 * update_slice( -1, .......,  iz-1 - RADIUS_H, .....)   for removal
 * update_slice( +1, .......,  iz   + RADIUS_H, .....)   for addition
 *
 * So the update function is where the optimised things occur.
 * To optimise the reading between neighbouring voxels of the working group
 * formed by the  WS_Y*WS_X  neighbouring voxels of a slice we read in
 * parallel the (RADIUS_H + WS_Y + RADIUS_H)(RADIUS_H + WS_X + RADIUS_H)
 * interesting voxels of a mutualised slice.
 * Once such interesting area is loaded in shared memory, all the threads
 * of the working group can access the interesting voxels of the slice
 * (iz-1 - RADIUS_H) for subtraction or addition.
 *
 * We use for this the variable SLICE which is declared as
 * extern __shared__ float SLICE[] ;
 *
 * which will be used to address the voxels of the loaded interesting
 * region.
 * But it is not over.
 * We need also to store the vector of histogram values.
 * Each thread keeps a vector of its histogram values
 * To do this we define
 *
 * #define myhisto(i)     SLICE[WW_Y*WW_X + (i)*blockDim.y*blockDim.x + tid ]
 *
 * This address the shared memory so that we don't interfere with the bottom
 * WW_Y*WW_X block which is dedicated to the loaded interesting regions, and
 * in a way such that myhisto(i) addresses a privatised vector which is
 * privatised for each thread tid.
 * The basic idea is that thread tid_a has all its histogram on a different
 * memory-bank than thread tid_b when tid_a is different from tid_b
 *
 * Memory banks for SLICE access are different when loading because i_in_tile
 * is initialised to tid.
 *
 * They are also different when accessing SLICE because
 * float v = SLICE[ threadIdx.x+sx +RADIUS_H + WW_X * (threadIdx.y+sy+RADIUS_H) ];
 * and threadIdx.x varies over a range of width 16.
 * There could be a conflict when WW_X<16 or, if the hardware has warps of
 * 32, when WW_X>16 but not a multiple of 16
 */
__global__ void ISTOGRAMMA(
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
