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
      // iadd=1 --> aggiungi    iadd=-1 --> togli
      const int iadd,
      // il numero del thread all'interno del cuda block (  tid>=0  && tid< blockDimY*blockDimX   )
      const int tid,
      // posizione thread all interno del blocco
      const int tiy, const int tix,
      // le dimensioni del pezzo di slice che si legge
      //   WW_Y =   RADIUS_H+blockDimY+RADIUS_H ;
      //   WW_X =   RADIUS_H+blockDimX+RADIUS_H ;
      const int WW_Y, const int WW_X,
      // il raggio della zone di interesse intorno al voxel
      const int RADIUS_H,
      //  Le coordinate y,x del block's corner nel volume globale
      const int block_cy, const int block_cx,
      // la coordinata z della slice nel volume globale
      const int islice,
      // definizione istogramma
      const int NBINS,
      // dimensioni cuda block
      const int blockDimY, const int blockDimX,
      // input data
      const float *d_volume_in,
      // dimensioni slice
      const int NY, const int NX)
   {
   int i_in_tile = tid;
   while (i_in_tile < WW_Y * WW_X)
      {

      int global_x = block_cx + (i_in_tile % WW_X) - RADIUS_H;
      int global_y = block_cy + (i_in_tile / WW_X) - RADIUS_H;

      float res = -1;
      float val = 0; // padding to zero

      if( global_x >= 0 && global_x < NX && global_y >= 0 && global_y < NY )
         val = d_volume_in[global_x + NX * (global_y + NY * islice)];
      for(int i = 0; i < NBINS + 1; i++)
         if(val >= bins_limits[i])
            res = i;

      SLICE[i_in_tile] = res;
      i_in_tile += blockDimY *blockDimX;
      }

   __syncthreads();    // la slice e' caricata completamente

   for (int sx = -RADIUS_H; sx <= RADIUS_H; sx++)
      {
      for (int sy = -RADIUS_H; sy <= RADIUS_H; sy++)
         {
         const float v = SLICE[tix + sx + RADIUS_H + WW_X * (tiy + sy + RADIUS_H)];

         if (v >= 0 && v < NBINS)
            myhisto((int) v) += iadd;
         }
      }

   __syncthreads(); // la slice e' stata utilizzata. Dopo questo semaforo la si potra cambiare con la prossima.
   }

// FOR THE ZERO PADDING
__device__ void update_slice_with_zeros(
      // iadd=1 --> aggiungi    iadd=-1 --> togli
      const int iadd,
      // il numero del thread all'interno del cuda block (  tid>=0  && tid< blockDimY*blockDimX   )
      const int tid,
      // posizione thread all interno del blocco
      const int tiy, const int tix,
      // le dimensioni del pezzo di slice che si legge
      //   WW_Y =   RADIUS_H+blockDimY+RADIUS_H ;
      //   WW_X =   RADIUS_H+blockDimX+RADIUS_H ;
      const int WW_Y, const int WW_X,
      // il raggio della zone di interesse intorno al voxel
      const int RADIUS_H,
      //  Le coordinate y,x del block's corner nel volume globale
      const int block_cy, const int block_cx, // NOT USED for padding
      // la coordinata z della slice nel volume globale
      const int islice, // NOT USED for padding
      // definizione istogramma
      const int NBINS,
      // dimensioni cuda block
      const int blockDimY, const int blockDimX,
      // input data
      const float *d_volume_in,
      // dimensioni slice
      const int NY, const int NX)
   {
   int i_in_tile = tid;
   while (i_in_tile < WW_Y * WW_X)
      {
      float res = -1;
      float val = 0; // padding to zero

      for (int i = 0; i < NBINS + 1; i++)
         {
         if (val >= bins_limits[i])
            res = i;
         }

      SLICE[i_in_tile] = res;
      i_in_tile += blockDimY * blockDimX;
      }

   __syncthreads();    // la slice e' caricata completamente

   for (int sx = -RADIUS_H; sx <= RADIUS_H; sx++)
      {
      for (int sy = -RADIUS_H; sy <= RADIUS_H; sy++)
         {
         const float v = SLICE[tix + sx + RADIUS_H + WW_X * (tiy + sy + RADIUS_H)];

         if (v >= 0 && v < NBINS)
            myhisto((int) v) += iadd;
         }
      }

   __syncthreads(); // la slice e' stata utilizzata. Dopo questo semaforo la si potra cambiare con la prossima.
   }

/*  =======================================
 * ISTOGRAMMA: Kernel principale
 *
 * Istogramma is launched with a 2D grid of 2D blocks.
 * The dimensions of each block are (from slow (Y) to fast (X) )  WS_Y, WS_X
 * A block, then, works on a tile of a 2D slice, and, inside the kernel,
 * there is a loop on the Z dimension,  named iz inside ISTOGRAMMA.
 *
 * Always inside ISTOGRAMMA, the block and grid coordinates are translated
 * to iy,ix
 * which are the planar coordinates of a voxel. The z coordinate is
 * represented by
 * the loop variable iz.
 *
 * Now, imagine that for a given voxel iz,iy,ix you know the voxel
 * histogram.
 * Then the variable iz increase by one.
 *
 * How do we obtain the histogram for voxel (iz,iy,ix) when we know already
 * the histogram for voxel (iz-1,iy,ix)?
 * We remove the contribution from slice  iz-1 - RADIUS_H
 * the we add the contribution from slice iz + RADIUS_H
 *
 * This is done by calling
 * update_slice( -1, .......,  iz-1 - RADIUS_H, .....)   for removal
 * update_slice( +1, .......,  iz   + RADIUS_H, .....)   for addition
 *
 * (Post NOTE : now with zero padding I have duplicated to
 * update_slice_with_zeros
 *   to be used for non existing slices. There is probably one more
 * efficient way to do
 * it but it was the fastest to code)
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
 * float v = SLICE[ tix+sx +RADIUS_H + WW_X * (tiy+sy+RADIUS_H) ];
 * and tix varies over a range of width 16.
 * There could be a conflict when WW_X<16 or, if the hardware has warps of
 * 32, when WW_X>16 but not a multiple of 16
 */
__global__ void ISTOGRAMMA(
      // il volume target di dimensioni (slow to fast )  NZ,NY, NBINS, NX
      float *d_volume_histo,
      // // definizione istogramma
      const int NBINS,
      // volume input di dimensioni (slow to fast)  NZ, NY, NX
      const float *d_volume_in,
      const int NX, const int NY, const int NZ,
      // il raggio della zone di interesse intorno al voxel
      const int RADIUS_H)
   {
   const int WW_Y = RADIUS_H + blockDim.y + RADIUS_H;
   const int WW_X = RADIUS_H + blockDim.x + RADIUS_H;

   const int tix = threadIdx.x;
   const int tiy = threadIdx.y;

   const int bidx = blockIdx.x;
   const int bidy = blockIdx.y;

   const int block_cx = bidx * blockDim.x;
   const int block_cy = bidy * blockDim.y;

   const int ix = block_cx + tix;
   const int iy = block_cy + tiy;

   const int tid = tiy * blockDim.x + tix;

   // every thread clears its histogram in shared memory
   for (int i = 0; i < NBINS; i++)
      myhisto(i)= 0;

   // prologo
   for (int islice = 0; islice < 2 * RADIUS_H + 1; islice++)
      update_slice_with_zeros(+1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H, block_cy,
            block_cx, 0, NBINS, blockDim.y, blockDim.x, d_volume_in, NY, NX);

   for (int islice = 0; islice < RADIUS_H; islice++)
      {
      if (islice < NZ)
         {
         update_slice_with_zeros(-1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H,
               block_cy, block_cx, 0, NBINS, blockDim.y, blockDim.x,
               d_volume_in, NY, NX);
         update_slice(+1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H, block_cy,
               block_cx, islice, NBINS, blockDim.y, blockDim.x, d_volume_in, NY,
               NX);
         }
      }
   // logo
   for (int iz = 0; iz < NZ; iz++)
      {
      if (iz - 1 - RADIUS_H >= 0)
         update_slice(-1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H, block_cy,
               block_cx, iz - 1 - RADIUS_H, NBINS, blockDim.y, blockDim.x,
               d_volume_in, NY, NX);
      else
         update_slice_with_zeros(-1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H,
               block_cy, block_cx, 0, NBINS, blockDim.y, blockDim.x,
               d_volume_in, NY, NX);
      if (iz + RADIUS_H < NZ)
         update_slice(+1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H, block_cy,
               block_cx, iz + RADIUS_H, NBINS, blockDim.y, blockDim.x,
               d_volume_in, NY, NX);
      else
         update_slice_with_zeros(+1, tid, tiy, tix, WW_Y, WW_X, RADIUS_H,
               block_cy, block_cx, 0, NBINS, blockDim.y, blockDim.x,
               d_volume_in, NY, NX);
      for (int ibin = 0; ibin < NBINS; ibin++)
         {
         if (ix < NX && iy < NY)
            d_volume_histo[ibin + NBINS * (ix + (long int) NX * (iy + NY * iz))] =
                  myhisto(ibin);  // non-coalesced writing
         }
      }
   }
