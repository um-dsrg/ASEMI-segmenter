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

        if( global_x >= 0 && global_x < NX && global_y >= 0 && global_y < NY ) {
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
                // d_volume_histo[ ix  + NX*( iy + (long int) NY*( ibin + NBINS*iz ) ) ] = myhisto(ibin);  // scrittura cohalesced
                d_volume_histo[ ibin + NBINS*( ix + (long int) NX*( iy + NY*iz ) ) ] = myhisto(ibin);  // scrittura NON cohalesced
        }
    }
}

#define TILE_DIM 16
#define TILE_HEIGHT 4
__global__ void myTranspose(float *odata, const float *idata, int NZ, int NY , int  NX) {
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
