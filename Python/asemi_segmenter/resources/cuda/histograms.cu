/* CUDA 3D and 2D neighbourhood histograms
 *
 * 3D histogram originally written by Alessandro Mirone for the ASEMI project.
 * Integrated and updated by Johann A. Briffa
 * All other code written by Johann A. Briffa
 */

#include <stdint.h>

/* Dynamic shared memory region
 *
 * The first WW_X * WW_Y elements hold the tile neighbourhood, where:
 *    WW_X = (radius + blockDim.x + radius)
 *    WW_Y = (radius + blockDim.y + radius)
 *
 * The next blockDim.x * blockDim.y * num_bins elements hold the histograms for
 * each thread (i.e. a separate histogram for every voxel in the tile).
 *
 * In the case of 2D histograms, x,y above are replaced by the two dimensions
 * defining the plane in which histograms are taken (i.e. the neighbouring
 * dimensions).
 */

extern __shared__ int8_t shared_memory[];

/* Computes histogram contribution from the tiles on the x,y plane at global_z.
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
__device__ void update_tiles_xy(
      // iadd = +1/-1 to add/subtract
      const int iadd,
      // thread index in CUDA block ( tid >= 0 && tid < blockDim.x * blockDim.y )
      const int tid,
      // size of the tile read in this block
      const int WW_X, const int WW_Y,
      // neighbourhood radius around voxel for computing histogram
      const int radius,
      // x,y coordinates of block's corner in global volume
      const int block_cx, const int block_cy,
      // z coordinate of thread's voxel in global volume
      const int global_z,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ)
   {
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_X * WW_Y];
   // parallel read of tile into shared memory
   for (int i_in_tile = tid;
         i_in_tile < WW_X * WW_Y;
         i_in_tile += blockDim.x * blockDim.y)
      {
      // determine x,y coordinates for this thread
      const int global_x = block_cx + (i_in_tile % WW_X) - radius;
      const int global_y = block_cy + (i_in_tile / WW_X) - radius;
      // read value at x,y,z from global memory if exists, zero otherwise
      ${data_t} val = 0;
      if( global_x >= 0 && global_x < NX &&
          global_y >= 0 && global_y < NY &&
          global_z >= 0 && global_z < NZ )
         val = d_volume_in[global_x + NX * (global_y + NY * global_z)];
      // convert value to corresponding histogram bin index
      // and store in shared memory
      shared_tile[i_in_tile] = int(num_bins * (val - min_range) / float(max_range - min_range));
      }
   // make sure all slice section is read
   __syncthreads();
   // each thread computes histogram for neighbourhood of its coordinate
   for (int sx = -radius; sx <= radius; sx++)
      for (int sy = -radius; sy <= radius; sy++)
         {
         // get neighbouring pixel bin index
         const int v = int(shared_tile[threadIdx.x + sx + radius + WW_X * (threadIdx.y + sy + radius)]);
         // add to histogram if within limits
         if (v >= 0 && v < num_bins)
            shared_hist[v * blockDim.x * blockDim.y + tid] += iadd;
         }
   // make sure all slice section is processed
   __syncthreads();
   }

/* Computes histogram contribution from the tiles on the x,z plane at global_y.
 * (Auxiliary Function)
 *
 * See update_tiles_xy for more information.
 */
__device__ void update_tiles_xz(
      // iadd = +1/-1 to add/subtract
      const int iadd,
      // thread index in CUDA block ( tid >= 0 && tid < blockDim.x * blockDim.z )
      const int tid,
      // size of the tile read in this block
      const int WW_X, const int WW_Z,
      // neighbourhood radius around voxel for computing histogram
      const int radius,
      // x,z coordinates of block's corner in global volume
      const int block_cx, const int block_cz,
      // y coordinate of thread's voxel in global volume
      const int global_y,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ)
   {
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_X * WW_Z];
   // parallel read of tile into shared memory
   for (int i_in_tile = tid;
         i_in_tile < WW_X * WW_Z;
         i_in_tile += blockDim.x * blockDim.z)
      {
      // determine x,z coordinates for this thread
      const int global_x = block_cx + (i_in_tile % WW_X) - radius;
      const int global_z = block_cz + (i_in_tile / WW_X) - radius;
      // read value at x,y,z from global memory if exists, zero otherwise
      ${data_t} val = 0;
      if( global_x >= 0 && global_x < NX &&
          global_y >= 0 && global_y < NY &&
          global_z >= 0 && global_z < NZ )
         val = d_volume_in[global_x + NX * (global_y + NY * global_z)];
      // convert value to corresponding histogram bin index
      // and store in shared memory
      shared_tile[i_in_tile] = int(num_bins * (val - min_range) / float(max_range - min_range));
      }
   // make sure all slice section is read
   __syncthreads();
   // each thread computes histogram for neighbourhood of its coordinate
   for (int sx = -radius; sx <= radius; sx++)
      for (int sz = -radius; sz <= radius; sz++)
         {
         // get neighbouring pixel bin index
         const int v = int(shared_tile[threadIdx.x + sx + radius + WW_X * (threadIdx.z + sz + radius)]);
         // add to histogram if within limits
         if (v >= 0 && v < num_bins)
            shared_hist[v * blockDim.x * blockDim.z + tid] += iadd;
         }
   // make sure all slice section is processed
   __syncthreads();
   }

/* Computes histogram contribution from the tiles on the y,z plane at global_x.
 * (Auxiliary Function)
 *
 * See update_tiles_xy for more information.
 */
__device__ void update_tiles_yz(
      // iadd = +1/-1 to add/subtract
      const int iadd,
      // thread index in CUDA block ( tid >= 0 && tid < blockDim.y * blockDim.z )
      const int tid,
      // size of the tile read in this block
      const int WW_Y, const int WW_Z,
      // neighbourhood radius around voxel for computing histogram
      const int radius,
      // y,z coordinates of block's corner in global volume
      const int block_cy, const int block_cz,
      // x coordinate of thread's voxel in global volume
      const int global_x,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ)
   {
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_Y * WW_Z];
   // parallel read of tile into shared memory
   for (int i_in_tile = tid;
         i_in_tile < WW_Y * WW_Z;
         i_in_tile += blockDim.x * blockDim.z)
      {
      // determine y,z coordinates for this thread
      const int global_y = block_cy + (i_in_tile % WW_Y) - radius;
      const int global_z = block_cz + (i_in_tile / WW_Y) - radius;
      // read value at x,y,z from global memory if exists, zero otherwise
      ${data_t} val = 0;
      if( global_x >= 0 && global_x < NX &&
          global_y >= 0 && global_y < NY &&
          global_z >= 0 && global_z < NZ )
         val = d_volume_in[global_x + NX * (global_y + NY * global_z)];
      // convert value to corresponding histogram bin index
      // and store in shared memory
      shared_tile[i_in_tile] = int(num_bins * (val - min_range) / float(max_range - min_range));
      }
   // make sure all slice section is read
   __syncthreads();
   // each thread computes histogram for neighbourhood of its coordinate
   for (int sy = -radius; sy <= radius; sy++)
      for (int sz = -radius; sz <= radius; sz++)
         {
         // get neighbouring pixel bin index
         const int v = int(shared_tile[threadIdx.y + sy + radius + WW_Y * (threadIdx.z + sz + radius)]);
         // add to histogram if within limits
         if (v >= 0 && v < num_bins)
            shared_hist[v * blockDim.y * blockDim.z + tid] += iadd;
         }
   // make sure all slice section is processed
   __syncthreads();
   }

/* Computes the 3D neighbourhood histogram for a given range of voxels.
 * (Main Kernel)
 *
 * Kernel is launched with a 2D grid of 2D blocks, with a thread for each voxel
 * in a single slice (XY plane). Each block covers a 2D tile, so that threads
 * in a block cache the tile neighbourhood in shared memory. A range of slices
 * are considered with an internal loop over the Z dimension. For each voxel
 * (ix,iy,iz) we need to compute the neighbourhood histogram.
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
      ${result_t} *d_volume_histo,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ,
      // range of slices to consider in each dimension (half-open)
      const int x_start, const int x_stop,
      const int y_start, const int y_stop,
      const int z_start, const int z_stop,
      // neighbourhood radius around voxel for computing histogram
      const int radius)
   {
   // size of the tile read in this block
   const int WW_X = 2 * radius + blockDim.x;
   const int WW_Y = 2 * radius + blockDim.y;
   // x,y coordinates of block's corner in global volume
   const int block_cx = x_start + blockIdx.x * blockDim.x;
   const int block_cy = y_start + blockIdx.y * blockDim.y;
   // thread index in CUDA block ( tid >= 0 && tid < blockDim.x * blockDim.y )
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
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_X * WW_Y];

   // every thread clears its histogram in shared memory
   for (int i = 0; i < num_bins; i++)
      shared_hist[i * blockDim.x * blockDim.y + tid]  = 0;

   // computation of first slice
   for (int z_offset = -radius; z_offset <= radius; z_offset++)
      {
      update_tiles_xy(+1, tid, WW_X, WW_Y, radius, block_cx, block_cy,
            z_start + iz + z_offset,
            min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
      }
   // copy histogram from shared memory to global memory (non-coalesced)
   for (int ibin = 0; ibin < num_bins; ibin++)
      {
      if (ix < HX && iy < HY)
         d_volume_histo[ibin + num_bins * (ix + HX * (iy + HY * iz))] =
               shared_hist[ibin * blockDim.x * blockDim.y + tid];
      }
   // computation of following slices
   for (iz++; iz < HZ; iz++)
      {
      update_tiles_xy(-1, tid, WW_X, WW_Y, radius, block_cx, block_cy,
            z_start + iz - 1 - radius,
            min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
      update_tiles_xy(+1, tid, WW_X, WW_Y, radius, block_cx, block_cy,
            z_start + iz + radius,
            min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
      // copy histogram from shared memory to global memory (non-coalesced)
      for (int ibin = 0; ibin < num_bins; ibin++)
         {
         if (ix < HX && iy < HY)
            d_volume_histo[ibin + num_bins * (ix + HX * (iy + HY * iz))] =
                  shared_hist[ibin * blockDim.x * blockDim.y + tid];
         }
      }
   }

/* Computes the 2D neighbourhood (XY plane) histogram for a given range of voxels.
 * (Main Kernel)
 *
 * Kernel is launched with a 3D grid of 2D blocks, with a thread for each voxel
 * (ix,iy,iz) in the volume, for which we need to compute the neighbourhood
 * histogram. Each block covers a 2D tile on the plane for which the
 * neighbourhood is defined, so that threads in a block cache the tile
 * neighbourhood in shared memory.
 *
 * Storage is always in row-major order, so from slow to fast the dimensions
 * are Z, Y, X, or slice, row, column.
 *
 * The histograms of all voxels in the volume are computed in parallel, with
 * the computation for voxels in the same block benefiting from the use of
 * shared memory.
 */
__global__ void histogram_2d_xy(
      // histogram output matrix
      ${result_t} *d_volume_histo,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ,
      // range of slices to consider in each dimension (half-open)
      const int x_start, const int x_stop,
      const int y_start, const int y_stop,
      const int z_start, const int z_stop,
      // neighbourhood radius around voxel for computing histogram
      const int radius)
   {
   // size of the tile read in this block
   const int WW_X = 2 * radius + blockDim.x;
   const int WW_Y = 2 * radius + blockDim.y;
   // x,y coordinates of block's corner in global volume
   const int block_cx = x_start + blockIdx.x * blockDim.x;
   const int block_cy = y_start + blockIdx.y * blockDim.y;
   // thread index in CUDA block ( tid >= 0 && tid < blockDim.x * blockDim.y )
   const int tid = threadIdx.y * blockDim.x + threadIdx.x;
   // size of the histogram output matrix
   const int HX = x_stop - x_start;
   const int HY = y_stop - y_start;
   const int HZ = z_stop - z_start;
   // x,y,z coordinates of thread's voxel in histogram output matrix
   const int ix = block_cx + threadIdx.x - x_start;
   const int iy = block_cy + threadIdx.y - y_start;
   const int iz = blockIdx.z;
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_X * WW_Y];

   // every thread clears its histogram in shared memory
   for (int i = 0; i < num_bins; i++)
      shared_hist[i * blockDim.x * blockDim.y + tid]  = 0;

   // computation
   update_tiles_xy(+1, tid, WW_X, WW_Y, radius, block_cx, block_cy,
         z_start + iz,
         min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
   // copy histogram from shared memory to global memory (non-coalesced)
   for (int ibin = 0; ibin < num_bins; ibin++)
      {
      if (ix < HX && iy < HY)
         d_volume_histo[ibin + num_bins * (ix + HX * (iy + HY * iz))] =
               shared_hist[ibin * blockDim.x * blockDim.y + tid];
      }
   }

/* Computes the 2D neighbourhood (XZ plane) histogram for a given range of voxels.
 * (Main Kernel)
 *
 * See histogram_2d_xy for more information.
 */
__global__ void histogram_2d_xz(
      // histogram output matrix
      ${result_t} *d_volume_histo,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ,
      // range of slices to consider in each dimension (half-open)
      const int x_start, const int x_stop,
      const int y_start, const int y_stop,
      const int z_start, const int z_stop,
      // neighbourhood radius around voxel for computing histogram
      const int radius)
   {
   // size of the tile read in this block
   const int WW_X = 2 * radius + blockDim.x;
   const int WW_Z = 2 * radius + blockDim.z;
   // x,z coordinates of block's corner in global volume
   const int block_cx = x_start + blockIdx.x * blockDim.x;
   const int block_cz = z_start + blockIdx.z * blockDim.z;
   // thread index in CUDA block ( tid >= 0 && tid < blockDim.x * blockDim.z )
   const int tid = threadIdx.z * blockDim.x + threadIdx.x;
   // size of the histogram output matrix
   const int HX = x_stop - x_start;
   const int HY = y_stop - y_start;
   const int HZ = z_stop - z_start;
   // x,y,z coordinates of thread's voxel in histogram output matrix
   const int ix = block_cx + threadIdx.x - x_start;
   const int iy = blockIdx.y;
   const int iz = block_cz + threadIdx.z - z_start;
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_X * WW_Z];

   // every thread clears its histogram in shared memory
   for (int i = 0; i < num_bins; i++)
      shared_hist[i * blockDim.x * blockDim.z + tid]  = 0;

   // computation
   update_tiles_xz(+1, tid, WW_X, WW_Z, radius, block_cx, block_cz,
         y_start + iy,
         min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
   // copy histogram from shared memory to global memory (non-coalesced)
   for (int ibin = 0; ibin < num_bins; ibin++)
      {
      if (ix < HX && iz < HZ)
         d_volume_histo[ibin + num_bins * (ix + HX * (iy + HY * iz))] =
               shared_hist[ibin * blockDim.x * blockDim.z + tid];
      }
   }

/* Computes the 2D neighbourhood (YZ plane) histogram for a given range of voxels.
 * (Main Kernel)
 *
 * See histogram_2d_xy for more information.
 */
__global__ void histogram_2d_yz(
      // histogram output matrix
      ${result_t} *d_volume_histo,
      // histogram definition
      const int min_range, const int max_range, const int num_bins,
      // input volume and size
      const ${data_t} *d_volume_in,
      const int NX, const int NY, const int NZ,
      // range of slices to consider in each dimension (half-open)
      const int x_start, const int x_stop,
      const int y_start, const int y_stop,
      const int z_start, const int z_stop,
      // neighbourhood radius around voxel for computing histogram
      const int radius)
   {
   // size of the tile read in this block
   const int WW_Y = 2 * radius + blockDim.y;
   const int WW_Z = 2 * radius + blockDim.z;
   // y,z coordinates of block's corner in global volume
   const int block_cy = y_start + blockIdx.y * blockDim.y;
   const int block_cz = z_start + blockIdx.z * blockDim.z;
   // thread index in CUDA block ( tid >= 0 && tid < blockDim.y * blockDim.z )
   const int tid = threadIdx.z * blockDim.y + threadIdx.y;
   // size of the histogram output matrix
   const int HX = x_stop - x_start;
   const int HY = y_stop - y_start;
   const int HZ = z_stop - z_start;
   // x,y,z coordinates of thread's voxel in histogram output matrix
   const int ix = blockIdx.x;
   const int iy = block_cy + threadIdx.y - y_start;
   const int iz = block_cz + threadIdx.z - z_start;
   // pointers for shared-memory regions
   ${index_t} *shared_tile = (${index_t} *)(shared_memory);
   ${result_t} *shared_hist = (${result_t} *)&shared_tile[WW_Y * WW_Z];

   // every thread clears its histogram in shared memory
   for (int i = 0; i < num_bins; i++)
      shared_hist[i * blockDim.y * blockDim.z + tid]  = 0;

   // computation
   update_tiles_yz(+1, tid, WW_Y, WW_Z, radius, block_cy, block_cz,
         x_start + ix,
         min_range, max_range, num_bins, d_volume_in, NX, NY, NZ);
   // copy histogram from shared memory to global memory (non-coalesced)
   for (int ibin = 0; ibin < num_bins; ibin++)
      {
      if (iy < HY && iz < HZ)
         d_volume_histo[ibin + num_bins * (ix + HX * (iy + HY * iz))] =
               shared_hist[ibin * blockDim.y * blockDim.z + tid];
      }
   }
