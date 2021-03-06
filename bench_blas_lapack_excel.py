#!/usr/bin/env python


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
#import cPickle as pickle
import pickle
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
    for i in range(trials):
        #t_start = time.time()
        result = np.linalg.qr(A)
        #times.append(time.time() - t_start)

    toc = time.time()-tic

    for i in range(trials):
        t_start = time.time()
        result = np.linalg.qr(A)
        times.append(time.time() - t_start)


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
    #final_times = [0, 0, 0, 0]

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4./3.)*N*N*N*1e-9, final_times


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
    for i in range(trials):
        #t_start = time.time()
        result = np.linalg.eig(A)
        #times.append(time.time() - t_start)
    
    toc = time.time()-tic

    for i in range(trials):
        t_start = time.time()
        result = np.linalg.eig(A)
        times.append(time.time() - t_start)
    
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
    #final_times = [0, 0, 0, 0]

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4./3.)*N*N*N*1e-9, final_times

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
    for i in range(trials):
        #t_start = time.time()
        #result = np.linalg.svd(A)
        result = np.linalg.svd(A, full_matrices=False)
        #times.append(time.time() - t_start)
 
    toc = time.time()-tic

    for i in range(trials):
        t_start = time.time()
        #result = np.linalg.svd(A)
        result = np.linalg.svd(A, full_matrices=False)
        times.append(time.time() - t_start)
 

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
    #final_times = [0, 0, 0, 0]

    #from math import exp
    #gflop = (4/3)*N*N*N*1e-9
    #print("N =", N)
    #print("N*N*N ", N*N*N)
    #print("N(N*N*1e-9 ", N*N*N*1e-9, N*N*N*exp(1e-9)) 
    #print("gflop ", gflop)
    #print("total_time/trials = ", toc/trials)
    #print("final times", final_times)

    #return toc/trials, (4/3)*N*N*N*1e-9
    return toc/trials, (4./3.)*N*N*N*1e-9, final_times
        
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
    for i in range(trials):
        #t_start = time.time()
        #result = np.linalg.svd(A)
        result = np.linalg.det(A)
        #times.append(time.time() - t_start)

    toc = time.time()-tic

    for i in range(trials):
        t_start = time.time()
        result = np.linalg.det(A)
        times.append(time.time() - t_start)


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
    #final_times = [0, 0, 0, 0]

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
    for i in range(trials):
        result = np.dot(a, a)  # - np.eye(N)
    
    toc = time.time()-tic # - mytic
    if gcold:
        gc.enable()

    
    return 1.0*toc/(trials), 2.0*N*N*N*1e-9


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
    for i in range(trials):
        #t_start = time.time()
        C = A.dot(B)
        #times.append(time.time() - t_start)

    toc = time.time()-tic

    for i in range(trials):
        t_start = time.time()
        C = A.dot(B)
        times.append(time.time() - t_start)


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
    #final_times = [0, 0, 0, 0]

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
    #print("CHOLESKY NATIVE CALL ==== %.5f" % total_cholesky)

    tic = time.time()
    times = []
    for i in range(trials):
        #t_start = time.time()
        L = np.linalg.cholesky(A)
        #times.append(time.time() - t_start)
        
    toc = time.time() - tic

    for i in range(trials):
        t_start = time.time()
        L = np.linalg.cholesky(A)
        times.append(time.time() - t_start)

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
    #final_times = [0, 0, 0, 0]

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
    for i in range(trials):
        #t_start = time.time()
        result = np.linalg.inv(data)
        #times.append(time.time() - t_start)
    toc = time.time()-tic
    
    times = []
    for i in range(trials):
        t_start = time.time()
        result = np.linalg.inv(data)
        times.append(time.time() - t_start)


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
    #final_times = [0, 0, 0, 0]

    #return toc/trials, 2*N*N*N*1e-9
    return toc/trials, 2.*N*N*N*1e-9, final_times


def time_numexpr(N=100, trials=5, dtype=np.double):
    #NOTE: This is giving me none sense results. At this moment I am not using this test

    x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    #x = np.arange(N)
    #y = np.arange(N)
    z = np.empty_like(x)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    times = []
    for i in range(trials):
        #t_start = time.time()
        #ne.evaluate('2*y+4*x', out = z)
        ne.evaluate('x*y - 4.1*x > 2.5*y', out = z)
        #times.append(time.time() - t_start)
    toc = time.time()-tic
    if gcold:
        gc.enable()

    #calculate average time and min time and also keep track of outliers (max time in the loop)
    #all_times = np.asarray(times)
    #min_time = np.amin(all_times)
    #max_time = np.amax(all_times)
    #mean_time = np.mean(all_times)
    #stdev_time = np.std(all_times)

    #print("Min = %.5f, Max = %.5f, Mean = %.5f, stdev = %.5f " % (min_time, max_time, mean_time, stdev_time))
    #final_times = [min_time, max_time, mean_time, stdev_time]
    final_times = [0, 0, 0, 0]

    #return (toc/trials, dtype().itemsize*3*N*1e-9)
    return toc/trials, 3.*N*1e-9, final_times

def test_timers():
    N = 500
    trials = 3
    dtype = np.double
    s, gflop = time_dgemm(N, trials, dtype)
    print("DGEMM   : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_cholesky(N, trials, dtype)
    print("Cholesky: N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    #s, gbyte = time_numexpr(50000, trials, dtype)
    #print("NumExpr : N: %d s: %e GBytes/s: %e" % (N, s, gbyte/s))


def bench(test_fun, Ns, trials, dtype=None):
    data = np.empty((len(Ns),2))
    gflops_comparison = np.empty((len(Ns),3))

    exectimes = np.empty((len(Ns),2))
    times_dict = dict()
    #print("%d tests" % len(Ns)
    tic = time.time()
    for i in range(len(Ns)):
        #sys.stdout.write('.')
        print('N=',Ns[i])
        sys.stdout.flush()
        if dtype is not None:
            out_tuple = test_fun(Ns[i],trials,dtype)
        else:
            out_tuple = test_fun(Ns[i],trials)
        #tuple contains: time/iterations, GFLOP, array with (min_time, max_time, avergare, std_dev)
        print("out_tuple", out_tuple)


        if len(out_tuple) > 1:
            #print('Return is more than 1 element')
            #print('Time for single operation = %f [s] when using %d trials (total time/trials [s])' % (out_tuple[0], trials))
            #print('alll times', out_tuple)
            #print('GFLOPS using total time %f', out_tuple[1]/out_tuple[0])
            
            #print('GFLOPS using min Time', out_tuple[1]/out_tuple[2][0])
            #print('GFLOPS using average Time', out_tuple[1]/out_tuple[2][2])
            
            #gflops for min time in itereation
            gflops_min_time = out_tuple[1]/out_tuple[2][0]
            #gflops for average time of all iterations
            gflops_tot_time_average = out_tuple[1]/out_tuple[0]
            
            data[i,:] = (Ns[i], out_tuple[1]/out_tuple[0])
          
            gflops_comparison[i,:] = (Ns[i], gflops_min_time, gflops_tot_time_average)
            
            #print('comparison',gflops_comparison)
         
            #dictionary: {Array_size : (Time_benchmark, [all_times])}
            times_dict[Ns[i]] = (out_tuple[0], np.asarray(out_tuple[2]))
        else:
            #print('Retrun is only 1 element', out_tuple)

            data[i,:] = (Ns[i], out_tuple[0])
            #exectimes[i,:] = (Ns[i], out_tuple[2])
            times_dict[Ns[i]] = (out_tuple[0], np.asarray(out_tuple[1]))
    #print('data', data)
    print('done')
    toc = time.time() - tic
    print('tests took: %4.4f seconds' % toc)
    
    #print(gflops_comparison)

    #print("data = \n", data)
    #print("times_disct = \n", times_dict)
    #print("gflosp_comparison = \n", gflops_comparison)

    #return data, exectimes
    return data, times_dict, gflops_comparison

def dump_data(data, data_dir, backend, algo, threads):
    filename = backend + '-' + algo + '-' + str(threads) + '.pkl'
    out_pickle = os.path.join(data_dir, filename)
    with open(out_pickle,'w') as data_file:
        pickle.dump(data, data_file)


def print_data(Ns, inputdata, execution_times, benchmark, backend, threads):
    #print("inputdata",inputdata)
    #print("exec", execution_times)
    my_Ns = []
    bench_time = []
    rest_times = []
    for key, value in execution_times.items():
        #print(key,value)
        my_Ns.append(key)
        bench_time.append(value[0])
        rest_times.append(value[1])
    my_Ns = sorted(my_Ns)
    #print("sorted list", my_Ns)
        
    """
    import pdb
    pdb.set_trace()
    print(my_Ns)
    for n in my_Ns:
        print(n)
        print(execution_times[n])

    print('bench_time',bench_time)
    print('NS =', my_Ns)
    print('execution_times ',execution_times)
    print('inputdata', inputdata)

    print('rest_times ',rest_times)
    #execution_times = inputdata[1]
    #bench_time = inputdata[0]
    #print("execution times", excution_times)
    """

    #i = 0
    outfilename = backend + '-' + benchmark + '-' + threads + '-' + 'times.txt' 
    with open(outfilename, 'w') as data_file:
        data_file.write("#Threads = %s , Python Distro = %s = " % (threads, pydistro))
        data_file.write('#Benchmark,  Matrix Size, Time per exec [s],    GFLOPs  \n' )
        #for size in Ns:
        i=0
        for size in sorted(execution_times):

            print('Benchmark = %s, Size = %d, Average GFlop/sec = %.5f' %  (benchmark, size, inputdata[i][1]))
            #final_times = [min_time, max_time, mean_time, stdev_time]
            if len(my_Ns) == 1:

                #data_file.write("#Threads = %s , Python Distro = %s = " % (threads, pydistro))

                #data_file.write("#Python Distro = %s" % pydistro)
                #data_file.write('#Benchmark,  Matrix Size,  Min Time,    Max Time,     Mean Time,    Stdev Time,  GFlops  \n' )
                #data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))

                #data_file.write('#Benchmark,  Matrix Size, Time per exec [s],    GFLOPs  \n' )
                
                #data_file.write('%s, %15d, %14.5f, %11.5f  \n' %  (benchmark, size, bench_time[i],  inputdata[i][1]))
                data_file.write('%s, %15d, %14.5f, %11.5f  \n' %  (benchmark, size, bench_time[0],  inputdata[0][1]))
            else:
                #print("xxx",execution_times,execution_times[size])
                #data_file.write('Benchmark = %15s, Size = %5d,Min Time %.5f, Max Time = %.5f, Mean Time = %.5f, Stdev Time = %.5f, GFlops = %.5f \n' %  (benchmark, size, execution_times[i][0],  execution_times[i][1],  execution_times[i][2],  execution_times[i][3], inputdata[i][1]))
                #data_file.write('%s, %15d, %14.5f, %11.5f,  %11.5f,  %11.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0],  execution_times[Ns[i]][1],  execution_times[Ns[i]][2],  execution_times[Ns[i]][3], inputdata[i][1]))
                #data_file.write('%s, %15d, %14.5f, %11.5f \n' %  (benchmark, size, execution_times[Ns[i]][0], inputdata[i][1]))
                data_file.write('%s, %15d, %14.5f, %11.5f \n' %  (benchmark, size, execution_times[size][0], inputdata[i][1]))
            i = i + 1


def create_spreadsheet(excel_filename):

    full_name = excel_filename + '.xlsx'
    # Create an new Excel file and add a worksheet.                                                                                                                                
    workbook = xlsxwriter.Workbook(full_name)
    worksheet = workbook.add_worksheet()

    return workbook, worksheet

def close_spreadsheet(workbook):
    workbook.close()

def make_spreadsheet(Ns, inputdata, gflops_comparison, benchmark, execution_times, backend, threads, row, workbook, worksheet, mkl, kmp):


    #print(inputdata)
    #print("xxxx",Ns, inputdata,benchmark)
    
    #gflops_times = [Size array, gflops_min_time, gflops_tot_time_average]
    #wehre gflops_min_time is the performance calculated using min time from set of iterations
    #gflops_tot_time_average is performance calculated using average of all times in iterations
    my_Ns = []
    bench_time = []
    rest_times = []
    for key, value in execution_times.items():
        #print(key,value)
        my_Ns.append(key)
        bench_time.append(value[0])
        rest_times.append(value[1])
        
    #print(bench_time)
    #print(rest_times)
    #rest_times order: Min Time, Max Time, Average, Stddev

    # Add a bold format to use to highlight cells.                                                                                                                                 
    bold = workbook.add_format({'bold': True})
    worksheet.set_column('A:A', 15)
    worksheet.set_column('C:C', 20)
    worksheet.set_column('H:H', 20)
    worksheet.set_column('I:I', 20)
    worksheet.set_column('J:J', 20)
    # Write some simple text.                                                                                                                                                      
    if row == 0:
        worksheet.write(0, 0, 'MKL_NUM_THREADS', bold)
        worksheet.write(0, 1, mkl , bold)
        worksheet.write(0, 2, 'KMP_AFFINITY', bold)
        worksheet.write(0, 3, kmp, bold)
        worksheet.write(1, 0, 'Benchmark', bold)
        worksheet.write(1, 1, 'Matrix Size', bold)
        worksheet.write(1, 2, 'Time for one operation [s]', bold)
        worksheet.write(1, 3, 'Min Time [s]', bold)
        worksheet.write(1, 4, 'Max Time [s]', bold)
        worksheet.write(1, 5, 'Average Time [s]', bold)
        worksheet.write(1, 6, 'Stddev [s]', bold)
        worksheet.write(1, 7, 'GFLOPs (tot time/trials)', bold)
        worksheet.write(1, 8, 'GFLOPs (average time for all trials)', bold)
        worksheet.write(1, 9, 'GFLOPs (Min time for all trials)', bold)
        worksheet.write(2, 0, benchmark)
        worksheet.write(2, 1, Ns)
        worksheet.write(2, 2, bench_time[0])
        worksheet.write(2, 3, rest_times[0][0])
        worksheet.write(2, 4, rest_times[0][1])
        worksheet.write(2, 5, rest_times[0][2])
        worksheet.write(2, 6, rest_times[0][3])
        worksheet.write(2, 7, inputdata[0][1])
        worksheet.write(2, 8, gflops_comparison[0][2])
        worksheet.write(2, 9, gflops_comparison[0][1])
    else:
        worksheet.write(row+2, 0, benchmark)
        worksheet.write(row+2, 1, Ns)
        worksheet.write(row+2, 2, bench_time[0])
        worksheet.write(row+2, 3, rest_times[0][0])
        worksheet.write(row+2, 4, rest_times[0][1])
        worksheet.write(row+2, 5, rest_times[0][2])
        worksheet.write(row+2, 6, rest_times[0][3])
        worksheet.write(row+2, 7, inputdata[0][1])
        worksheet.write(row+2, 8, gflops_comparison[0][2])
        worksheet.write(row+2, 9, gflops_comparison[0][1])


    if row == 4:
        workbook.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Benchmark for NumPy Matrix Multiplication, Cholesky, LU and Single Value decomposition.")
    parser.add_argument('--pydistro', required=True, help="Use 'intel' for Intel distribution\n 'accelerate' for Anaconda Accelerate\n 'anaconda' for Anaconda")
    parser.add_argument('--threads', required=True, help="Set environment MKL_NUM_THREADS env variable")
    parser.add_argument('--filename', required=False, help="Excel spreadsheet file name")
    parser.add_argument('-s', '--svd', action='store_true', help='Run SVD')
    parser.add_argument('-q', '--qr', action='store_true', help='Run QR')
    parser.add_argument('-i', '--inv', action='store_true', help='Run Matrix Inversion')
    parser.add_argument('-d', '--dgemm', action='store_true', help='Run DGEMM')
    parser.add_argument('-c', '--cholesky', action='store_true', help='Run Cholesky')
    parser.add_argument('-a','--all', action='store_true', help="Run all benchmarks")
    parser.add_argument('--debug', action='store_true', help="Set this for Debugging/Testing (use small array sizes for all benchmarks")
    args = parser.parse_args()

    pydistro = args.pydistro
    numthreads = args.threads
    if args.filename is not None:
        excel_filename = args.filename
    debug = args.debug


    #Define lists
    Ns = []
    #inputdata = [dgemm_data, inv_data, qr_data, cholesky_data, svd_data]
    inputdata = []
    #benchmark = ['DGEMM', 'Inversion', 'QR', 'Cholesky', 'SVD']
    benchmark = []
    #execution_times = [execution_times_dgemm, execution_times_inv, execution_times_qr, execution_times_chol, execution_times_svd]
    execution_times = []
    #gflops_compare_list = [gflops_comparison_dgemm, gflops_comparison_inv, gflops_comparison_qr, gflops_comparison_chol, gflops_comparison_svd]
    gflops_compare_list = []    

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

    mkl_threads = os.environ['MKL_NUM_THREADS']
    kmp_affinity = os.environ['KMP_AFFINITY']
    pyVersion = sys.version

    print('Python Version =', pyVersion)
    print('MKL_NUM_THREADS = ', os.environ['MKL_NUM_THREADS'])
    print('KMP_AFFINITY = ', os.environ['KMP_AFFINITY'])


    #print("checking timers...")
    #test_timers()

    trials = 20
    dtype = np.double
    print('Number of iterations to use:', trials)
 
    #Array sizes for each benchmark
    #Use the following for Benchmarking

    if debug:
        #Set of arrays to use when modifying code for debuggin or testing purposes
        Ns_qr =  np.array([300,500,800,1000,1200,1400,1800])
        Ns_svd = np.array([300])
        Ns_dgemm = np.array([500])
        Ns_chol =  np.array([500,600])#,1500,2000,5000,10000,20000])
        Ns_inv = np.array([500])
    else:
        #Set of arrays to run official benchmarks
        #Ns_qr =  np.array([30000])
        #Ns_svd = np.array([3500])
        #Ns_dgemm = np.array([40000])
        #Ns_chol =  np.array([40000])
        #Ns_inv = np.array([25000])
        #QR
        Ns_qr = np.array([500,2000,4000, 8000,10000, 15000, 20000, 30000])
        #SVD
        Ns_svd = np.array([200,400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 8000, 10000])
        #Matrix Inversion
        Ns_inv = np.array([1000,2000,3000,4000, 6000, 8000, 10000,15000, 20000, 25000])    
        #DGEMM
        Ns_dgemm = np.array([192, 384, 768, 1920, 3840, 7680, 19200, 40000]) 
        #Cholesky
        Ns_chol = np.array([1000,2000, 3000, 4000, 8000, 10000, 15000, 20000,25000,30000, 40000])


    #inputdata = [dgemm_data, inv_data, qr_data, cholesky_data, svd_data]
    inputdata = []
    #benchmark = ['DGEMM', 'Inversion', 'QR', 'Cholesky', 'SVD']
    benchmark = []
    #execution_times = [execution_times_dgemm, execution_times_inv, execution_times_qr, execution_times_chol, execution_times_svd]
    execution_times = []
    #gflops_compare_list = [gflops_comparison_dgemm, gflops_comparison_inv, gflops_comparison_qr, gflops_comparison_chol, gflops_comparison_svd]


    if (args.svd or args.all) and (not args.debug):
        Ns.append(Ns_svd)
        benchmark.append('SVD')

        print('Benchmarking SVD')

        svd_data,execution_times_svd, gflops_comparison_svd = bench(test_svd, Ns_svd, trials, dtype)
        print_data(Ns_svd, svd_data, execution_times_svd, 'SVD', backend, numthreads)

        inputdata.append(svd_data)
        execution_times.append(execution_times_svd)
        gflops_compare_list.append(gflops_comparison_svd)
    
    #Run in debug mode for a specific benchmark
    if (args.debug and args.svd) or (args.debug and args.all):
        Ns.append(Ns_svd)
        benchmark.append('SVD')

        print('Benchmarking SVD')

        svd_data,execution_times_svd, gflops_comparison_svd = bench(test_svd, Ns_svd, trials, dtype)
        print_data(Ns_svd, svd_data, execution_times_svd, 'SVD', backend, numthreads)

        inputdata.append(svd_data)
        execution_times.append(execution_times_svd)
        gflops_compare_list.append(gflops_comparison_svd)

    if (args.qr or args.all) and (not args.debug):
        Ns.append(Ns_qr)
        benchmark.append('QR')
        print('benchmarking QR decomposition')

        qr_data, execution_times_qr, gflops_comparison_qr = bench(time_qr, Ns_qr, trials)
        #print('qr_data',qr_data)
        print_data(Ns_qr, qr_data, execution_times_qr, 'QR', backend, numthreads)

        inputdata.append(qr_data)
        execution_times.append(execution_times_qr)
        gflops_compare_list.append(gflops_comparison_qr)

    if (args.qr and args.debug) or (args.all and args.debug):
        Ns.append(Ns_qr)
        benchmark.append('QR')
        print('benchmarking QR decomposition')

        qr_data, execution_times_qr, gflops_comparison_qr = bench(time_qr, Ns_qr, trials)
        #print('qr_data',qr_data)
        print_data(Ns_qr, qr_data, execution_times_qr, 'QR', backend, numthreads)

        inputdata.append(qr_data)
        execution_times.append(execution_times_qr)
        gflops_compare_list.append(gflops_comparison_qr)

    if (args.inv or args.all) and (not args.debug):
        Ns.append(Ns_inv)
        benchmark.append('INV')
        print('benchmarking invertion Matrix')

        inv_data,execution_times_inv, gflops_comparison_inv = bench(time_inv, Ns_inv, trials)
        print_data(Ns_inv, inv_data, execution_times_inv, 'Invertion', backend, numthreads)

        inputdata.append(inv_data)
        execution_times.append(execution_times_inv)
        gflops_compare_list.append(gflops_comparison_inv)

    if (args.inv and args.debug) or (args.all and args.debug):
        Ns.append(Ns_inv)
        benchmark.append('INV')
        print('benchmarking invertion Matrix')

        inv_data,execution_times_inv, gflops_comparison_inv = bench(time_inv, Ns_inv, trials)
        print_data(Ns_inv, inv_data, execution_times_inv, 'Invertion', backend, numthreads)

        inputdata.append(inv_data)
        execution_times.append(execution_times_inv)
        gflops_compare_list.append(gflops_comparison_inv)

    if (args.dgemm or args.all) and (not args.debug):
        Ns.append(Ns_dgemm)
        benchmark.append('DGEMM')

        print('benchmarking DGEMM')

        dgemm_data, execution_times_dgemm, gflops_comparison_dgemm = bench(time_dgemm, Ns_dgemm, trials, dtype)
        print_data(Ns_dgemm, dgemm_data, execution_times_dgemm, 'DGEMM', backend, numthreads)

        inputdata.append(dgemm_data)
        execution_times.append(execution_times_dgemm)
        gflops_compare_list.append(gflops_comparison_dgemm)


    if (args.dgemm and args.debug) or (args.all and args.debug):
        Ns.append(Ns_dgemm)
        benchmark.append('DGEMM')

        print('benchmarking DGEMM')

        dgemm_data, execution_times_dgemm, gflops_comparison_dgemm = bench(time_dgemm, Ns_dgemm, trials, dtype)
        print_data(Ns_dgemm, dgemm_data, execution_times_dgemm, 'DGEMM', backend, numthreads)

        inputdata.append(dgemm_data)
        execution_times.append(execution_times_dgemm)
        gflops_compare_list.append(gflops_comparison_dgemm)


    if (args.cholesky or args.all) and (not args.debug):
        Ns.append(Ns_chol)
        benchmark.append('Cholesky')

        print('benchmarking Cholesky')

        cholesky_data, execution_times_chol, gflops_comparison_chol = bench(time_cholesky, Ns_chol, trials, dtype)
        print_data(Ns_chol, cholesky_data, execution_times_chol, 'Cholesky', backend, numthreads)

        inputdata.append(cholesky_data)
        execution_times.append(execution_times_chol)
        gflops_compare_list.append(gflops_comparison_chol)


    if (args.cholesky and args.debug) or (args.all and args.debug):
        Ns.append(Ns_chol)
        benchmark.append('Cholesky')

        print('benchmarking Cholesky in debug mode (small array)')

        cholesky_data, execution_times_chol, gflops_comparison_chol = bench(time_cholesky, Ns_chol, trials, dtype)
        print_data(Ns_chol, cholesky_data, execution_times_chol, 'Cholesky', backend, numthreads)

        inputdata.append(cholesky_data)
        execution_times.append(execution_times_chol)
        gflops_compare_list.append(gflops_comparison_chol)



    #if excel_filename:

    """
    try:
        import xlsxwriter
    except ImportError:
        print('Install xlsxwriter package to be able to create an output excel spreadsheet')
        write_spreadsheet(excel_filename)
    workbook, worksheet = create_spreadsheet(excel_filename)

    for i in range(len(Ns)):

        make_spreadsheet(Ns[i], inputdata[i], gflops_compare_list[i], benchmark[i], execution_times[i], backend, numthreads, i, workbook, worksheet, mkl_threads, kmp_affinity)

    close_spreadsheet(workbook)
    """

    #If running a set of array sizes for each of the benchmarks, these are recommended.
    #Be aware some of them will take a long time if set to use 1 thread.
    
    #QR
    #Ns_qr = np.array([500,2000,4000, 8000,10000, 15000, 20000, 30000])
    #SVD
    #Ns_svd = np.array([200,400, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 8000, 10000])
    #Matrix Inversion
    #Ns_inv = np.array([1000,2000,3000,4000, 6000, 8000, 10000,15000, 20000, 25000])    
    #DGEMM
    #Ns_dgemm = np.array([192, 384, 768, 1920, 3840, 7680, 19200, 40000]) 
    #Cholesky
    #Ns_chol = np.array([1000,2000, 3000, 4000, 8000, 10000, 15000, 20000,25000,30000, 40000])

    """
    print('benchmarking QR decomposition')

    qr_data, execution_times_qr, gflops_comparison_qr = bench(time_qr, Ns_qr, trials)
    print('qr_data',qr_data)
    print_data(Ns_qr, qr_data, execution_times_qr, 'QR', backend, numthreads)
    #dump_data(qr_data, data_dir, backend, 'QR', numthreads)
          
    print('benchmarking SVD')

    svd_data,execution_times_svd, gflops_comparison_svd = bench(test_svd, Ns_svd, trials, dtype)
    print_data(Ns_svd, svd_data, execution_times_svd, 'SVD', backend, numthreads)
    #dump_data(svd_data, data_dir, backend, 'SVD', numthreads)

 
    print('benchmarking invertion Matrix')

    inv_data,execution_times_inv, gflops_comparison_inv = bench(time_inv, Ns_inv, trials)
    print_data(Ns_inv, inv_data, execution_times_inv, 'Invertion', backend, numthreads)
    #dump_data(inv_data, data_dir, backend, 'Invertion', numthreads)

    
    print('benchmarking DGEMM')

    dgemm_data, execution_times_dgemm, gflops_comparison_dgemm = bench(time_dgemm, Ns_dgemm, trials, dtype)
    print_data(Ns_dgemm, dgemm_data, execution_times_dgemm, 'DGEMM', backend, numthreads)
    #dump_data(dgemm_data, data_dir, backend, 'DGEMM', numthreads)


    print('benchmarking Cholesky')

    cholesky_data, execution_times_chol, gflops_comparison_chol = bench(time_cholesky, Ns_chol, trials, dtype)
    print_data(Ns_chol, cholesky_data, execution_times_chol, 'Cholesky', backend, numthreads)
    #dump_data(cholesky_data, data_dir, backend, 'Cholesky', numthreads)
    

    #setup of list to create spreadsheet
    #Ns = [Ns_dgemm, Ns_inv, Ns_qr, Ns_chol, Ns_svd]
    Ns = [Ns_qr, Ns_chol]
    #inputdata = [dgemm_data, inv_data, qr_data, cholesky_data, svd_data]
    inputdata = [qr_data, cholesky_data]
    #benchmark = ['DGEMM', 'Inversion', 'QR', 'Cholesky', 'SVD']
    benchmark = ['QR', 'Cholesky']
    #execution_times = [execution_times_dgemm, execution_times_inv, execution_times_qr, execution_times_chol, execution_times_svd]
    execution_times = [execution_times_qr, execution_times_chol]
    #gflops_compare_list = [gflops_comparison_dgemm, gflops_comparison_inv, gflops_comparison_qr, gflops_comparison_chol, gflops_comparison_svd]
    gflops_compare_list = [gflops_comparison_qr, gflops_comparison_chol]
    """



    """
    #NumExpr is producing wierd results. Need to understand what is happening
    print('benchmarking NumExpr')
    #logNs = np.arange(12, 18.5, 0.5) # uncomment to run big tests
    logNs = np.arange(12, 20.5, 0.5) # uncomment to run big tests
    #logNs = np.arange(6,13.5,0.5) # uncomment to run quick tests
    Ns = np.exp2(logNs)
    
    #Ns = np.array([1e7])
    numexpr_data,execution_times = bench(time_numexpr, Ns, trials, dtype)
    print_data(Ns, numexpr_data, execution_times, 'NumExpres', backend, numthreads)
    dump_data(numexpr_data, data_dir, backend, 'NumExpr',numthreads)

    """
