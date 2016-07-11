#
# http://software.intel.com/en-us/intel-mkl
# https://code.google.com/p/numexpr/wiki/NumexprVML

from __future__ import print_function
import datetime
import sys
from scipy import stats
import numpy as np
#import numexpr as ne
import time
import gc
import os.path
import cPickle as pickle
import os
import argparse

data_dir = './'

def time_qr(N=100, trials=3, dtype=np.double):
    """
    Test QR decomposition
    """
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()

    #warm up run for MKL
    result = np.linalg.qr(A)

    tic = time.time()
    times = []
    for i in xrange(trials):
        t_start = time.time()
        result = np.linalg.qr(A)
        times.append(time.time() - t_start)

    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4/3)*N*N*N*1e-9, final_times


def test_eigenvalue(N=100, trials=3, dtype=np.double):
    """
    Test eigen value computation of a matrix
    """
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()
    
    #warm up run for MKL
    result = np.linalg.eig(A)

    tic = time.time()
    times = []
    for i in xrange(trials):
        t_start = time.time()
        result = np.linalg.eig(A)
        times.append(time.time() - t_start)
    
    toc = time.time()-tic
    if gcold:
        gc.enable()


    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4/3)*N*N*N*1e-9, final_times

def test_svd(N=100, trials=3, dtype=np.double):
    """
    Test single value decomposition of a matrix
    """
    #i = 2000
    #data = random((i,i))
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()

    #warm up run for MKL
    result = np.linalg.svd(A, full_matrices=False)

    tic = time.time()    
    times = []
    for i in xrange(trials):
        t_start = time.time()
        #result = np.linalg.svd(A)
        result = np.linalg.svd(A, full_matrices=False)
        times.append(time.time() - t_start)
 
    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4/3)*N*N*N*1e-9, final_times
        
def test_det(N=100, trials=3, dtype=np.double):
    """
    Test the computation of the matrix determinant
    """

    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()
    singleTicToc = []

    #warm up run for MKL
    result = np.linalg.det(A)

    tic = time.time()    
    times = []
    for i in xrange(trials):
        t_start = time.time()
        #result = np.linalg.svd(A)
        result = np.linalg.det(A)
        times.append(time.time() - t_start)

    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, N*N*N*1e-9
    return toc/trials, N*N*N*1e-9, final_times

def time_dot(N=100, trials=3, dtype=np.double):
    """                                                                                                         
    Test the dot product (vector vector multiplication)                                                                                        
    """
    a = np.asarray(np.random.rand(N), dtype=dtype)
    #b = np.linalg.inv(a)
    #b = np.asarray(np.random.rand(N), dtype=dtype)

    gcold = gc.isenabled()
    gc.disable()

    tic = time.time()
    times = []
    for i in xrange(trials):
        result = np.dot(a, a)  # - np.eye(N)
    
    toc = time.time()-tic # - mytic
    if gcold:
        gc.enable()

    
    return 1.0*toc/(trials), 2*N*N*N*1e-9


def time_dgemm(N=100, trials=3, dtype=np.double):
    
    LARGEDIM = 20000
    KSIZE = N
    A = np.asarray(np.random.rand(LARGEDIM, KSIZE), dtype=dtype)
    B = np.asarray(np.random.rand(KSIZE, LARGEDIM), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()

    # warm up run for MKL
    C = A.dot(B)

    tic = time.time()
    times = []
    for i in xrange(trials):
        t_start = time.time()
        C = A.dot(B)
        times.append(time.time() - t_start)

    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return 1.*toc/trials, 2E-9*LARGEDIM*LARGEDIM*KSIZE
    return 1.*toc/trials, 2E-9*LARGEDIM*LARGEDIM*KSIZE, final_times

def time_cholesky(N=100, trials=5, dtype=np.double):
    
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    A = A*A.transpose() + N*np.eye(N)
    gcold = gc.isenabled()
    gc.disable()

    #warm up run for MKL
    L = np.linalg.cholesky(A)
    
    #get the time it take to run cholesky
    chol_time = time.time()
    L = np.linalg.cholesky(A)
    total_cholesky = time.time() - chol_time
    print("CHOLESKY NATIVE CALL ==== %.5f" % total_cholesky)

    tic = time.time()
    times = []
    for i in xrange(trials):
        t_start = time.time()
        L = np.linalg.cholesky(A)
        times.append(time.time() - t_start)

    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, N*N*N/3.0*1e-9
    return toc/trials, N*N*N/3.0*1e-9, final_times

def time_inv(N=100, trials=5, dtype=np.double):

    data = np.asarray(np.random.rand(N,N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()

    #warm up run for MKl
    result = np.linalg.inv(data)


    tic = time.time()
    times = []
    for i in xrange(trials):
        t_start = time.time()
        result = np.linalg.inv(data)
        times.append(time.time() - t_start)

    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    all_times = np.asarray(times)
    min_time = np.amin(all_times)
    max_time = np.amax(all_times)
    mean_time = np.mean(all_times)
    stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    final_times = [min_time, max_time, mean_time, stdev_time]

    #return toc/trials, 2*N*N*N*1e-9
    return toc/trials, 2*N*N*N*1e-9, final_times


def time_numexpr(N=100, trials=5, dtype=np.double):
    #NOTE: This is giving me none sense results. At this moment I am not using this test

    #x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    #y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    x = np.arange(N)
    y = np.arange(N)
    z = np.empty_like(x)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in xrange(trials):
        #ne.evaluate('2*y+4*x', out = z)
        ne.evaluate('x*y - 4.1*x > 2.5*y', out = z)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    #return (toc/trials, dtype().itemsize*3*N*1e-9)
    return toc/trials, 3*N*1e-9

def test_timers():
    N = 500
    trials = 3
    dtype = np.double
    s, gflop = time_dgemm(N, trials, dtype)
    print("DGEMM   : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_cholesky(N, trials, dtype)
    print("Cholesky: N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gbyte = time_numexpr(50000, trials, dtype)
    print("NumExpr : N: %d s: %e GBytes/s: %e" % (N, s, gbyte/s))


def bench(test_fun, Ns, trials, dtype=None):
    data = np.empty((len(Ns),2))
    exectimes = np.empty((len(Ns),2))
    times_dict = dict()
    #print("%d tests" % len(Ns)
    tic = time.time()
    for i in xrange(len(Ns)):
        #sys.stdout.write('.')
        print('N=',Ns[i])
        sys.stdout.flush()
        if dtype is not None:
            out_tuple = test_fun(Ns[i],trials,dtype)
        else:
            out_tuple = test_fun(Ns[i],trials)

        if len(out_tuple) > 1:
            data[i,:] = (Ns[i], out_tuple[1]/out_tuple[0])
            times_dict[Ns[i]] = np.asarray(out_tuple[2])
        else:
            data[i,:] = (Ns[i], out_tuple[0])
            #exectimes[i,:] = (Ns[i], out_tuple[2])
            times_dict[Ns[i]] = np.asarray(out_tuple[1])
    print('data', data)
    print('done')
    toc = time.time() - tic
    print('tests took: %e seconds' % toc)
    
    #return data, exectimes
    return data, times_dict

def dump_data(data, data_dir, backend, algo, threads):
    filename = backend + '-' + algo + '-' + str(threads) + '.pkl'
    out_pickle = os.path.join(data_dir, filename)
    with open(out_pickle,'w') as data_file:
        pickle.dump(data, data_file)


def print_data(Ns, inputdata, execution_times, benchmark, backend, threads):
    print("inputdata",inputdata)
    i = 0
    outfilename = backend + '-' + benchmark + '-' + threads + '-' + 'times.txt' 
    with open(outfilename, 'w') as data_file:
        for size in Ns:
            print('Benchmark = %s, Size = %d, Average GFlop/sec = %.5f' %  (benchmark, size, inputdata[i][1]))
            #final_times = [min_time, max_time, mean_time, stdev_time]
            if i == 0:
                data_file.write("#Threads = %s" % threads)
                data_file.write("#Python Distro = %s" % pydistro)
                data_file.write('#Benchmark,  Matrix Size,  Min Time,    Max Time,     Mean Time,    Stdev Time,  GFlops  \n' )
                data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))
            else:
                #data_file.write('Benchmark = %15s, Size = %5d,Min Time %.5f, Max Time = %.5f, Mean Time = %.5f, Stdev Time = %.5f, GFlops = %.5f \n' %  (benchmark, size, execution_times[i][0],  execution_times[i][1],  execution_times[i][2],  execution_times[i][3], inputdata[i][1]))
                data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))
            i = i + 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Benchmark for NumPy Matrix Multiplication, Cholesky, LU and Single Value decomposition.")
    parser.add_argument('--pydistro', required=True, help="Use 'intel' for Intel distribution\n 'accelerate' for Anaconda Accelerate\n 'anaconda' for Anaconda")
    parser.add_argument('--threads', required=True, help="Set environment MKL_NUM_THREADS env variable")
    args = parser.parse_args()

    pydistro = args.pydistro
    numthreads = args.threads
 
    """
    #add to the path the distro
    if pydistro == 'intel':
        myalias = 'pyintel27'
    elif pydistro == 'anaconda':
        myalias = 'pyana27'
    elif pydistro == 'accelerate':
        myalias = 'pyanamkl'

    print('setting up in path %s' % myalias)
    os.system(myalias)
    """

    print("before import mkl")
    try:
        import mkl
        have_mkl = True
        backend = pydistro
        print("Running with MKL Acceleration")
    except ImportError:
        have_mkl = False
        myPythonVersion = sys.version
        if 'Anaconda' in myPythonVersion:
            backend = 'Anaconda'
        elif 'Anaconda' not in myPythonVersion:
            backend = pydistro
        print("Running with normal backends")

    #Set parameters for MKL and OMP
    
    #os.environ["MKL_NUM_THREADS"] = args.threads
    #os.environ["MKL_DYNAMICS"] = 'FALSE'
    #os.environ["OMP_NUM_THREADS"] = '1'
  
    #if numthreads == 32:
    #    os.environ['MKL_NUM_THREADS'] = '32'
    #else:
    #    os.environ['MKL_NUM_THREADS'] = '1'

    print('MKL_NUM_THREADS = ', os.environ['MKL_NUM_THREADS'])
    print('KMP_AFFINITY = ', os.environ['KMP_AFFINITY'])


    print("checking timers...")
    #test_timers()
    #logNs = np.arange(6,13.5,0.5) # uncomment to run the big stuff (orginal)
    #logNs = np.arange(6,15.5,0.5) # uncomment to run the big stuff
    #logNs = np.arange(3,7,0.5) # uncomment to run quick tests
    #Ns = np.exp2(logNs) #(original)
    #Ns = np.array([20000,25000])
    
    #print(Ns)
    #import sys
    #sys.exit()
    trials = 3
    dtype = np.double
    print('Number of iterations:', trials)
 
    #print('benchmarking Determinant')
    #Ns = np.array([23000])
    #det_data = bench(test_det, Ns, trials)
    #dump_data(det_data, data_dir, backend, 'Determinant')


    print('benchmarking QR decomposition')
    #logNs = np.arange(6,13.5,0.5) # uncomment to run the big stuff
    #Ns = np.exp2(logNs) #(original)
    #Ns = np.array([10000, 15000, 20000])
    #Ns = np.array([500,2000,4000, 8000,10000, 15000, 20000, 30000])
    #For PSF
    Ns =  np.array([30000])
    #Ns =  np.array([100])
    qr_data, execution_times = bench(time_qr, Ns, trials)
    print_data(Ns, qr_data, execution_times, 'QR', backend, numthreads)
    dump_data(qr_data, data_dir, backend, 'QR', numthreads)
          


    #print('benchmarking eigenvalues')
    #Ns = np.array([7000])
    #logNs = np.arange(6,14.5,0.5) # uncomment to run the big stuff
    #Ns = np.exp2(logNs) #(original)
    #eigen_data = bench(test_eigenvalue, Ns, trials)
    #dump_data(eigen_data, data_dir, backend, 'Eigenvalues', numthreads)


    print('benchmarking SVD')
    #Ns = np.array([200,400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 8000, 10000])
    #Ns = np.array([200])
    #For PSF
    Ns = np.array([3500])#,4000])
    #logNs = np.arange(6,14.5,0.5) # uncomment to run the big stuff
    #Ns = np.exp2(logNs) #(original)
    svd_data,execution_times = bench(test_svd, Ns, trials, dtype)
    print_data(Ns, svd_data, execution_times, 'SVD', backend, numthreads)
    dump_data(svd_data, data_dir, backend, 'SVD', numthreads)

 
    print('benchmarking invertion Matrix')
    #logNs = np.arange(6,14.5,0.5) # uncomment to run the big stuff
    #Ns = np.exp2(logNs) #(original)
    #Ns = np.array([1000,2000,3000,4000, 6000, 8000, 10000,15000, 20000, 25000])
    #for PSF
    Ns = np.array([25000])
    #Ns = np.array([3500])
    #Ns = np.array([1000,1500,2000,2500,3000,3500,4000, 6000, 8000, 10000,12000])
    #Ns = np.array([2000,3000,4000,5000,6000,7000])
    #Ns = np.array([20000])#,3000,4000,5000,6000,7000])
    inv_data,execution_times = bench(time_inv, Ns, trials)
    print_data(Ns, inv_data, execution_times, 'Invertion', backend, numthreads)
    dump_data(inv_data, data_dir, backend, 'Invertion', numthreads)

    
    #print('benchmarking Dot')
    #logNs = np.arange(6,26.5,0.5) # uncomment to run the big stuff
    #Ns = np.exp2(logNs) #(original)
    #Ns = np.array([25000])
    #dot_data = bench(time_dot, Ns, trials)
    #dump_data(dot_data, data_dir, backend, 'Dot_Vectors', numthreads)


    #print('Ns', Ns)
    print('benchmarking DGEMM')
    #Ns = np.array([192, 384, 768, 1920, 3840, 7680, 19200, 38400])
    #For PSF
    Ns = np.array([40000])
    #Ns = np.array([960,1920])
    #Ns = np.array([192, 384, 768, 960, 1920])
    #Ns = np.array([192, 384]) #, 768, 960, 1920])
    dgemm_data, execution_times = bench(time_dgemm, Ns, trials, dtype)
    print_data(Ns, dgemm_data, execution_times, 'DGEMM', backend, numthreads)
    dump_data(dgemm_data, data_dir, backend, 'DGEMM', numthreads)


    print('benchmarking Cholesky')
    #Ns = np.array([1000,2000,3000])
    #32 threads
    #Ns = np.array([1000,2000, 3000, 4000, 8000, 10000, 15000, 20000,25000,30000, 40000])
    #1 thread Intel
    #Ns = np.array([500])#,1000,1500,2000,3000,4000,5000,10000,15000,20000,25000])
    #PSF
    #Ns = np.array([500,1000,1500,2000,3000,4000,5000,10000,20000])
    #For PSF 
    Ns =  np.array([40000])
    #dtype = np.half
    cholesky_data, execution_times = bench(time_cholesky, Ns, trials, dtype)
    print_data(Ns, cholesky_data, execution_times, 'Cholesky', backend, numthreads)
    dump_data(cholesky_data, data_dir, backend, 'Cholesky', numthreads)
    
    

    #NumExpr is producing wierd results. Need to understand what is happening
    #print('benchmarking NumExpr')
    #logNs = np.arange(12, 18.5, 0.5) # uncomment to run big tests
    #logNs = np.arange(12, 18.5, 0.5) # uncomment to run big tests
    #logNs = np.arange(6,13.5,0.5) # uncomment to run quick tests
    #Ns = np.exp2(logNs)
    
    #Ns = np.array([1e7])
    #numexpr_data = bench(time_numexpr, Ns, trials, dtype)
    #dump_data(numexpr_data, data_dir, backend, 'NumExpr')


