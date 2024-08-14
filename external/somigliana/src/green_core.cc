#include "../include/somigliana/green_core.h"
#include "../include/somigliana/green_core.inl"

#include <Eigen/Dense>

extern "C" {

    void mvc_cpu(scalar_t *d_PHI,
                const scalar_t *d_V,
                const index_t  *d_cageF,
                const scalar_t *d_cageV,
                const index_t nv,
                const index_t ncf,
                const index_t ncv) {
        const unsigned int blocksize = 256;
        const unsigned int numBlocks = (nv+blocksize-1)/blocksize;
        // mvc_kernel<<< numBlocks, blocksize >>>
        //     (d_PHI, d_V, d_cageF, d_cageV, nv, ncf, ncv);

#pragma omp parallel for
        for(index_t idx = 0; idx < nv; ++idx){
            mvc_kernel(d_PHI,
                d_V,
                d_cageF,
                d_cageV,
                nv,
                ncf,
                ncv,
                idx);
        }
    }
    
    void green_cpu(scalar_t *d_PHIx,
                    scalar_t *d_PHIy,
                    scalar_t *d_PHIz,
                    scalar_t *d_PSI,
                    const scalar_t *d_V,
                    const index_t  *d_cageF,
                    const scalar_t *d_cageV,
                    const scalar_t *d_cageN,
                    const index_t nv,
                    const index_t ncf,
                    const index_t ncv,
                    const scalar_t *d_qp,
                    const scalar_t *d_qw,
                    const index_t nq) {
        // parallel through basis columns
        // const unsigned int blocksize = 256;
        // const unsigned int numBlocks = (ncf*nv+blocksize-1)/blocksize;
        // green_kernel<<< numBlocks, blocksize >>>
        //     (d_phix, d_phiy, d_phiz, d_psi,
        //      d_V, d_cageF, d_cageV, d_cageN, nv, ncf, ncv,
        //      d_qp, d_qw, nq);

#pragma omp parallel for
        for(index_t idx = 0; idx < ncf * nv; ++idx){
            green_kernel(d_PHIx,
                d_PHIy,
                d_PHIz,
                d_PSI,
                d_V,
                d_cageF,
                d_cageV,
                d_cageN,
                nv,
                ncf,
                ncv,
                d_qp,
                d_qw,
                nq,
                //idx, // thread_index,
                idx / ncf, // index,
                idx % ncf // f
                );
        }
    }

    void somig_cpu(const scalar_t nu,
                    scalar_t *d_PHIx,
                    scalar_t *d_PHIy,
                    scalar_t *d_PHIz,
                    scalar_t *d_PSI ,
                    const scalar_t *d_V,
                    const index_t  *d_cageF,
                    const scalar_t *d_cageV,
                    const scalar_t *d_cageN,
                    const index_t nv,
                    const index_t ncf,
                    const index_t ncv,
                    const scalar_t *d_qp,
                    const scalar_t *d_qw,
                    const index_t nq) {
        // parallel through basis entries
        // const unsigned int blocksize = 256;
        // const unsigned int numBlocks = (ncf*nv+blocksize-1)/blocksize;
        // somig_kernel<<< numBlocks, blocksize >>>
        //     (nu, d_PHIx, d_PHIy, d_PHIz, d_PSI,
        //      d_V, d_cageF, d_cageV, d_cageN, nv, ncf, ncv,
        //      d_qp, d_qw, nq);
#pragma omp parallel for
        for(index_t idx = 0; idx < ncf * nv; ++idx){
            somig_kernel(
                    nu,
                d_PHIx,
                d_PHIy,
                d_PHIz,
                d_PSI,
                d_V,
                d_cageF,
                d_cageV,
                d_cageN,
                nv,
                ncf,
                ncv,
                d_qp,
                d_qw,
                nq,
                //idx, // thread_index,
                idx / ncf, // index,
                idx % ncf // f
                );
        }   

    }

  // reduce phixyz to phi
    void green_post_cpu(scalar_t *d_PHI,
                        const scalar_t *d_PHIx,
                        const scalar_t *d_PHIy,
                        const scalar_t *d_PHIz,
                        const index_t  *d_cageF,
                        const index_t nv,
                        const index_t ncf,
                        const index_t ncv) {
    // parallel through basis columns
    // const unsigned int blocksize = 256;
    // const unsigned int numBlocks = (nv+blocksize-1)/blocksize;
    // green_kernel_post<<< numBlocks, blocksize >>>
    //     (d_phi, d_phix, d_phiy, d_phiz, d_cageF, nv, ncf, ncv);

#pragma omp parallel for
        for(index_t idx = 0; idx < nv; ++idx){
            green_kernel_post(d_PHI,
                                d_PHIx,
                                d_PHIy,
                                d_PHIz,
                                d_cageF,
                                nv,
                                ncf,
                                ncv,
                                idx);
        }
    
    } 

    // reduce PHIxyz to PHI
    void somig_post_cpu(scalar_t *d_PHI,
                        const scalar_t *d_PHIx,
                        const scalar_t *d_PHIy,
                        const scalar_t *d_PHIz,
                        const index_t  *d_cageF,
                        const index_t nv,
                        const index_t ncf) {
    // parallel through basis columns
//     const unsigned int blocksize = 256;
//     const unsigned int numBlocks = (nv+blocksize-1)/blocksize;
//     somig_kernel_post<<< numBlocks, blocksize >>>
//         (d_PHI, d_PHIx, d_PHIy, d_PHIz, d_cageF, nv, ncf);

#pragma omp parallel for
        for(index_t idx = 0; idx < nv; ++idx){
            somig_kernel_post(d_PHI,
                                d_PHIx,
                                d_PHIy,
                                d_PHIz,
                                d_cageF,
                                nv,
                                ncf,
                                idx);
        }
    }    
  


// TODO: Remove?
void mvc_compute(scalar_t *d_PHI,
               const scalar_t *d_V,
               const index_t  *d_cageF,
               const scalar_t *d_cageV,               
               const index_t nv,
               const index_t ncf,
               const index_t ncv){
#ifdef SOMIG_WITH_CUDA
  mvc_gpu(d_PHI,
              d_V,
              d_cageF,
              d_cageV,               
              nv,
              ncf,
              ncv);
#else
  mvc_cpu(d_PHI,
              d_V,
              d_cageF,
              d_cageV,               
              nv,
              ncf,
              ncv);
#endif
  }

void green_compute(scalar_t *d_phix,
                scalar_t *d_phiy,
                scalar_t *d_phiz,
                scalar_t *d_psi,
                const scalar_t *d_V,
                const index_t  *d_cageF,
                const scalar_t *d_cageV,
                const scalar_t *d_cageN,
                const index_t nv,
                const index_t ncf,
                const index_t ncv,
                const scalar_t *d_qp,
                const scalar_t *d_qw,
                const index_t nq){
#ifdef SOMIG_WITH_CUDA
  green_gpu(d_phix,
            d_phiy,
            d_phiz,
            d_psi,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);

#else
  green_cpu(d_phix,
            d_phiy,
            d_phiz,
            d_psi,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);
#endif
}

void green_post(scalar_t *d_phi,
                    const scalar_t *d_phix,
                    const scalar_t *d_phiy,
                    const scalar_t *d_phiz,
                    const index_t  *d_cageF,
                    const index_t nv,
                    const index_t ncf,
                    const index_t ncv){
#ifdef SOMIG_WITH_CUDA
  green_post_gpu(d_phi,
                  d_phix,
                  d_phiy,
                  d_phiz,
                  d_cageF,
                  nv,
                  ncf,
                  ncv);  
#else
  green_post_cpu(d_phi,
                  d_phix,
                  d_phiy,
                  d_phiz,
                  d_cageF,
                  nv,
                  ncf,
                  ncv); 
#endif                                            
}
  
void somig_compute(const scalar_t nu,
                scalar_t *d_PHIx,
                scalar_t *d_PHIy,
                scalar_t *d_PHIz,
                scalar_t *d_PSI ,
                const scalar_t *d_V,
                const index_t  *d_cageF,
                const scalar_t *d_cageV,
                const scalar_t *d_cageN,
                const index_t nv,
                const index_t ncf,
                const index_t ncv,
                const scalar_t *d_qp,
                const scalar_t *d_qw,
                const index_t nq
                ){
#ifdef SOMIG_WITH_CUDA
  somig_gpu(nu,
            d_PHIx,
            d_PHIy,
            d_PHIz,
            d_PSI ,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);   
#else
  somig_cpu(nu,
            d_PHIx,
            d_PHIy,
            d_PHIz,
            d_PSI ,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);                  
#endif // WITH_CUDA
}

void somig_post(scalar_t *d_PHI,
                const scalar_t *d_PHIx,
                const scalar_t *d_PHIy,
                const scalar_t *d_PHIz,
                const index_t  *d_cageF,
                const index_t nv,
                const index_t ncf){
#ifdef SOMIG_WITH_CUDA
  somig_post_gpu(d_PHI,
                    d_PHIx,
                    d_PHIy,
                    d_PHIz,
                    d_cageF,
                    nv,
                    ncf);
#else // WITH_CUDA
  somig_post_cpu(d_PHI,
                    d_PHIx,
                    d_PHIy,
                    d_PHIz,
                    d_cageF,
                    nv,
                    ncf);
#endif    
  }


/**/
}