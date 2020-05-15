'''Module for histogram related functions.'''

import os
import numpy as np
import fast_histogram
import sys
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from asemi_segmenter.lib import regions

#########################################
def histogram(array, num_bins, value_range):
    '''
    Compute a histogram from a range and number of bins.

    :param numpy.ndarray array: The array of values.
    :param int num_bins: The number of bins in the histogram.
    :param tuple value_range: A 2-tuple consisting of the minimum and maximum values to consider
        with the maximum not being inclusive.
    :return: A 1D histogram array.
    :rtype: numpy.ndarray
    '''
    return fast_histogram.histogram1d(array, num_bins, value_range)
    # Some times fast_histogram fails test cases because of a minor bug (https://github.com/astrofrog/fast-histogram/issues/45) so this may be useful during testing.
    # return np.histogram(array, num_bins, value_range)[0]

#########################################
def apply_histogram_to_all_neighbourhoods_in_slice_3d(array_3d, slice_index, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=slice(None), col_slice=slice(None)):
    '''
    Find a histograms of the values in the neighbourhood around every element
    in a volume slice.

    Given a slice of voxels in a volume, this function will find the histogram
    of values around every voxel in the slice. The histogram around a voxel is
    computed using a cube of a given radius centered around the voxel. Radius
    defined such that the side of the cube is radius + 1 + radius long.

    Histograms of neighbourhoods around every voxel in a slice can be computed
    efficiently using a rolling algorithm that reuses histograms in other
    neighbouring voxels. Consider if we had to use this function on a 2D
    image instead of a volume and using a 2D square neighbourhood instead
    of a cube. We'll be using this 3x3 image as an example:
        [a,b,c]
        [d,e,f]
        [g,h,i]
    The neighbourhood around the pixel at (0,0) with radius 1 has the
    following frequencies:
        PAD=5, a=1, b=1, c=0, d=1, e=1, f=0, g=0, h=0, i=0 => [5,1,1,0,1,1,0,0,0,0]
    The neighbourhood around (0,1) has values in common with the
    neighbourhood around (0,0) so we can avoid counting all the elements in
    this neighbourhood by instead counting only what has changed from the
    previous neighbourhood and update the frequencies with new information
    in the dropped and new columns:
        histogram_01 = histogram_00 - histogram([PAD,a,d]) + histogram([PAD,c,f])
    Likewise, the neighbourhood around (1,0) has values in common with (0,0)
    as well and can be calculated by subtracting the dropped row and adding
    the new column:
        histogram_10 = histogram_00 - histogram([PAD,PAD,PAD]) + histogram([g,h,i])
    This means that we only need to perform a full histogram count for the
    top left corner as everything else can be computed by rolling the values.

    For extracting histograms in 3D space, we will still be only processing
    a single slice but with adjacent slices for context. Given a 3D array A,
    the neighbourhood around A[r,c] (in the slice of interest) with radius R
    and neighbouring_dims d is hist(A[r,c], R, d). For each d, the following
    optimisations can be performed:
        hist(A[r,c], R, {0,1,2}) = hist(A[r-1,c], R, {0,1,2})
                                    - hist(A[r-1-R,c], R, {0,2})
                                    + hist(A[r+R,c], R, {0,2})
                          (also) = hist(A[r,c-1], R, {0,1,2})
                                    - hist(A[r,c-1-R], R, {0,1})
                                    + hist(A[r,c+R], R, {0,1})

    :param numpy.ndarray array_3d: The volume from which to extract the histograms.
    :param int slice_index: The index of the slice to use within the volume.
    :param int radius: The radius of the neighbourhood around each voxel.
    :param set neighbouring_dims: The set of dimensions to apply the neighbourhoods on.
    :param int min_range: The minimum range of the values to consider.
    :param int max_range: The maximum range of the values to consider, not included.
    :param int num_bins: The number of bins in the histograms.
    :param int pad: The pad value for values outside the array.
    :param slice row_slice: The range of rows in the slice to consider.
    :param slice col_slice: The range of columns in the slice to consider.
    :return: A 3D array where the first two dimensions are equal to the dimensions of the slice
        and the last dimension is the number of bins.
    :rtype: numpy.ndarray
    '''
    [ _, num_rows_in, num_cols_in ] = array_3d.shape
    row_slice = slice(
            row_slice.start if row_slice.start is not None else 0,
            row_slice.stop  if row_slice.stop is not None else num_rows_in
        )
    col_slice = slice(
            col_slice.start if col_slice.start is not None else 0,
            col_slice.stop  if col_slice.stop is not None else num_cols_in
        )
    num_rows_out = row_slice.stop - row_slice.start
    num_cols_out = col_slice.stop - col_slice.start

    result = np.empty([ num_rows_out, num_cols_out, num_bins ], np.float32)

    if neighbouring_dims == {0,1,2}:
        for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
            for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                if col_out == 0 and row_out == 0:
                    result[row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in), radius, {0,1,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histogram (top left corner).
                elif col_out == 0:
                    result[row_out, col_out, :] = (
                            result[row_out-1, col_out, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in-1-radius, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row face.
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in+radius, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Include effect of new row face.
                        )
                else:
                    result[row_out, col_out, :] = (
                            result[row_out, col_out-1, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in-1-radius), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column face.
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in+radius), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Include effect of new column face.
                        )

    elif neighbouring_dims == {0,1}:
        for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
            for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                if row_out == 0:
                    result[row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in), radius, {0,1}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histograms (top row)
                else:
                    result[row_out, col_out, :] = (
                            result[row_out-1, col_out, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in-1-radius, col_in), radius, {0}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in+radius, col_in), radius, {0}, pad), num_bins, (min_range, max_range)) #Include effect of new row
                        )

    elif neighbouring_dims == {0,2}:
        for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
            for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                if col_out == 0:
                    result[row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in), radius, {0,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histograms (left column)
                else:
                    result[row_out, col_out, :] = (
                            result[row_out, col_out-1, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in-1-radius), radius, {0}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in+radius), radius, {0}, pad), num_bins, (min_range, max_range)) #Include effect of new column
                        )

    elif neighbouring_dims == {1,2}:
        for (row_out, row_in) in enumerate(range(row_slice.start, row_slice.stop)):
            for (col_out, col_in) in enumerate(range(col_slice.start, col_slice.stop)):
                if col_out == 0 and row_out == 0:
                    result[row_out, col_out, :] = histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in), radius, {1,2}, pad), num_bins, (min_range, max_range)) #Get the only completely computed histogram (top left corner)
                elif col_out == 0:
                    result[row_out, col_out, :] = (
                            result[row_out-1, col_out, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in-1-radius, col_in), radius, {2}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped row
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in+radius, col_in), radius, {2}, pad), num_bins, (min_range, max_range)) #Include effect of new row
                        )
                else:
                    result[row_out, col_out, :] = (
                            result[row_out, col_out-1, :]
                            -
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in-1-radius), radius, {1}, pad), num_bins, (min_range, max_range)) #Undo effect of dropped column
                            +
                            histogram(regions.get_neighbourhood_array_3d(array_3d, (slice_index, row_in, col_in+radius), radius, {1}, pad), num_bins, (min_range, max_range)) #Include effect of new column
                        )

    else:
        raise NotImplementedError('Only neighbouring dimensions of {0,1}, {0,2}, {1,2}, and {0,1,2} implemented.')

    return result

#########################################
def gpu_apply_histogram_to_all_neighbourhoods_in_slice_3d(array_3d, slice_index, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=slice(None), col_slice=slice(None)):
    '''
    GPU implementation of apply_histogram_to_all_neighbourhoods_in_slice_3d
    '''

    # CUDA code
    mod = SourceModule("""
#define MAXBINLIMS_ 41
    int MAXBINLIMS = MAXBINLIMS_ ;

    __device__ __constant__ float bins_limits[MAXBINLIMS_];

    // ogni blocco legge un pezzo di slice nella memoria shared
    //
    // ogni thread tiene il suo histogramma nella memoria share
    // si usa la variabile SHARED

    extern __shared__ float SHARED[];

#define SLICE SHARED

    //  WW_Y e WW_X sono le dimensioni del pezzo di slice che si legge.
    // Al di sopra si pone anche l'istogramma privatizzato ( i e' l'indice dell'istogramma, tid e' il numero della thread)

#define myhisto(i)     SHARED[WW_Y*WW_X + (i)*blockDim.y*blockDim.x + tid ]

    // ========================================================================
    // update_slice : FUNZIONE AUSILIARIA CHIAMATA DAL KERNEL PRINCIPALE
    //

    __device__ void update_slice(
                                  // iadd=1 --> aggiungi    iadd=-1 --> togli
                                  int iadd ,

                                  // il numero del thread all'interno del cuda block (  tid>=0  && tid< blockDimY*blockDimX   )
                                  int tid  ,

                                  // posizione thread all interno del blocco
                                  int tiy, int tix,


                                  // le dimensioni del pezzo di slice che si legge
                                  //   WW_Y =   RADIUS_H+blockDimY+RADIUS_H ;
                                  //   WW_X =   RADIUS_H+blockDimX+RADIUS_H ;
                                  int WW_Y ,
                                  int WW_X ,

                                  // il raggio della zone di interesse intorno al voxel
                                  int RADIUS_H ,

                                  //  Le coordinate y,x del block's corner nel volume globale
                                  int block_cy ,
                                  int block_cx ,

                                  // la coordinata z della slice nel volume globale
                                  int islice   ,

                                  // // definizione istogramma
                                  // float bottom ,
                                  // float step   ,
                                  int NBINS   ,

                                  // dimensioni cuda block
                                  int blockDimY ,
                                  int blockDimX ,


                                  // input data
                                  float *d_volume_in,

                                  // dimensioni slice
                                  int NY,
                                  int NX
 ) {

          int i_in_tile = tid;
          while(i_in_tile< WW_Y*WW_X) {

              int global_x =block_cx + (i_in_tile % WW_X) - RADIUS_H;
              int global_y =block_cy + (i_in_tile / WW_X) - RADIUS_H ;

              float res = -1;

              float val=0;   // padding to zero

              if( global_x >= 0 && global_x < NX && global_y >= 0 && global_y < NY     ) {
                val =   d_volume_in[ global_x +NX*( global_y+ NY*islice)]  ;
              }
              for(int i=0; i<NBINS+1; i++) {
                  if( val>= bins_limits[i]) res=i;
              }

              SLICE[i_in_tile] = res;
              i_in_tile += blockDimY *blockDimX ;
          }

          __syncthreads();    // la slice e' caricata completamente


          for(int sx=-RADIUS_H; sx<=RADIUS_H; sx++) {
              for(int sy=-RADIUS_H ; sy<=RADIUS_H ; sy++) {

                 float v  = SLICE[     tix+sx +RADIUS_H + WW_X * (tiy+sy+RADIUS_H) ];

                 if(v >= 0 && v < NBINS ) {
                     myhisto( ((int) v ) ) += iadd;
                 }
              }
          }

          __syncthreads();      // la slice e' stata utilizzata. Dopo questo semaforo la si potra cambiare con la prossima.
    }
    __device__ void update_slice_with_zeros(   // FOR THE ZERO PADDING
                                  // iadd=1 --> aggiungi    iadd=-1 --> togli
                                  int iadd ,

                                  // il numero del thread all'interno del cuda block (  tid>=0  && tid< blockDimY*blockDimX   )
                                  int tid  ,

                                  // posizione thread all interno del blocco
                                  int tiy, int tix,


                                  // le dimensioni del pezzo di slice che si legge
                                  //   WW_Y =   RADIUS_H+blockDimY+RADIUS_H ;
                                  //   WW_X =   RADIUS_H+blockDimX+RADIUS_H ;
                                  int WW_Y ,
                                  int WW_X ,

                                  // il raggio della zone di interesse intorno al voxel
                                  int RADIUS_H ,

                                  //  Le coordinate y,x del block's corner nel volume globale
                                  int block_cy ,
                                  int block_cx ,

                                  // la coordinata z della slice nel volume globale
                                  int islice   , // not used for padding

                                  // // definizione istogramma
                                  // float bottom ,
                                  // float step   ,
                                  int NBINS   ,

                                  // dimensioni cuda block
                                  int blockDimY ,
                                  int blockDimX ,


                                  // input data
                                  float *d_volume_in,

                                  // dimensioni slice
                                  int NY,
                                  int NX
 ) {

          int i_in_tile = tid;
          while(i_in_tile< WW_Y*WW_X) {

              int global_x =block_cx + (i_in_tile % WW_X) - RADIUS_H;
              int global_y =block_cy + (i_in_tile / WW_X) - RADIUS_H ;

              float res = -1;

              float val=0;   // padding to zero

              for(int i=0; i<NBINS+1; i++) {
                  if( val>= bins_limits[i]) res=i;
              }

              SLICE[i_in_tile] = res;
              i_in_tile += blockDimY *blockDimX ;
          }

          __syncthreads();    // la slice e' caricata completamente


          for(int sx=-RADIUS_H; sx<=RADIUS_H; sx++) {
              for(int sy=-RADIUS_H ; sy<=RADIUS_H ; sy++) {

                 float v  = SLICE[     tix+sx +RADIUS_H + WW_X * (tiy+sy+RADIUS_H) ];

                 if(v >= 0 && v < NBINS ) {
                     myhisto( ((int) v ) ) += iadd;
                 }
              }
          }

          __syncthreads();      // la slice e' stata utilizzata. Dopo questo semaforo la si potra cambiare con la prossima.
    }


    //  =======================================
    // ISTOGRAMMA: Kernel principale
    __global__ void ISTOGRAMMA(
                                 // il volume target di dimensioni (slow to fast )  NZ,NY, NBINS, NX
                                 float *d_volume_histo,

                                 // // definizione istogramma
                                 int NBINS,
                                 // float bottom,
                                 // float step,

                                 // volume input di dimensioni (slow to fast)  NZ, NY, NX
                                 float * d_volume_in,
                                 int NX  ,
                                 int  NY ,
                                 int NZ  ,

                                 // il raggio della zone di interesse intorno al voxel
                                 int RADIUS_H

    ) {

      const int WW_Y = RADIUS_H+ blockDim.y  +RADIUS_H ;
      const int WW_X = RADIUS_H+ blockDim.x  +RADIUS_H ;


      const int tix = threadIdx.x;
      const int tiy = threadIdx.y;

      const int bidx = blockIdx.x;
      const int bidy = blockIdx.y;

      int block_cx = bidx* blockDim.x;
      int block_cy = bidy* blockDim.y;

      int ix = block_cx + tix ;
      int iy = block_cy + tiy ;

      const int tid =  tiy *  blockDim.x + tix;


      for( int i=0; i< NBINS; i++) {
         myhisto(i) = 0;
      }

      // prologo
      for(int islice = 0; islice< 2*RADIUS_H+1 ; islice++) {
          update_slice_with_zeros( +1, tid, tiy, tix,  WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , 0,  NBINS, blockDim.y , blockDim.x , d_volume_in, NY, NX ) ;
      }

      for(int islice = 0; islice< RADIUS_H ; islice++) {
        if(islice<NZ) {
          update_slice_with_zeros( -1, tid, tiy, tix,  WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , 0,  NBINS, blockDim.y , blockDim.x , d_volume_in, NY, NX ) ;
          update_slice( +1, tid, tiy, tix,  WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , islice, /* bottom, step,*/ NBINS, blockDim.y , blockDim.x , d_volume_in, NY, NX ) ;
        }
      }
      // logo
      for(int iz = 0; iz<NZ; iz++) {
        if(iz-1-RADIUS_H >=0 ) {
          update_slice( -1, tid, tiy, tix,WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , iz-1 - RADIUS_H ,/* bottom, step,*/ NBINS, blockDim.y , blockDim.x  , d_volume_in, NY, NX  ) ;
        } else {
          update_slice_with_zeros( -1, tid, tiy, tix,  WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , 0,  NBINS, blockDim.y , blockDim.x , d_volume_in, NY, NX ) ;
        }
        if(iz+RADIUS_H <NZ ) {
          update_slice( +1, tid, tiy, tix, WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , iz + RADIUS_H ,  /*bottom, step,*/ NBINS, blockDim.y , blockDim.x   , d_volume_in, NY, NX ) ;
        }else {
          update_slice_with_zeros( +1, tid, tiy, tix,  WW_Y, WW_X,  RADIUS_H , block_cy, block_cx , 0,  NBINS, blockDim.y , blockDim.x , d_volume_in, NY, NX ) ;
        }
        for(int ibin = 0; ibin < NBINS ; ibin++) {
            if(ix< NX && iy < NY )
                   // d_volume_histo[     ix  + NX*( iy + (long int) NY*( ibin + NBINS*iz ) ) ] = myhisto(ibin);  // scrittura cohalesced
                   d_volume_histo[     ibin + NBINS*( ix + (long int) NX*( iy + NY*iz ) ) ] = myhisto(ibin);  // scrittura NON cohalesced
        }
      }
    }


    #define TILE_DIM 16
    #define TILE_HEIGHT 4
    __global__ void myTranspose(float *odata, const float *idata, int NZ, int NY , int  NX)
    {
      __shared__ float tile[TILE_DIM * TILE_DIM];

      int CX = blockIdx.x * TILE_DIM     ;
      int CY = blockIdx.y * TILE_DIM  ;
      int CZ = blockIdx.z * TILE_HEIGHT  ;

      int x =    + threadIdx.x;
      int y =    + threadIdx.y;
      int z = CZ + threadIdx.z;

        if( CY+y<NY && CX+x<NX  && z<NZ    ) {
           tile[( TILE_DIM*(z-CZ)  +  y)*TILE_DIM + x ] = idata[((long int)( z*NY + (long int) CY+y))*NX + CX+x];
        }

       __syncthreads();


        if( CX+y<NX && CY+x<NY && z<NZ  ) {
          odata[  ((long int)( z*NX + (long int) CX+y))*NY + CY+x    ] = tile[( TILE_DIM*(z-CZ)  + x)*TILE_DIM + y];
        }
       __syncthreads();

     }
    """)

    return result
