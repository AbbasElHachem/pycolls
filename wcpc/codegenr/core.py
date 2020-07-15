'''
@author: Faizan-Uni-Stuttgart

'''

# import os
# import timeit
# import time
# from pathlib import Path


class CodeGenr:
    '''Generate a text file, given a format
    '''
    def __init__(self, tab='\t'):
        self.code = []
        self.tab = tab
        self.level = 0
        return
 
    def ind(self):
        '''Increase indentation level by one
        '''
        self.level = self.level + 1
        return

    def ded(self, emp=True, lev=1):
        '''Decrease indentation level
        '''
        if self.level == 0:
            raise SyntaxError('Level is zero already!')
        self.level = self.level - lev
        
        if emp:
            self.els()
        return

    def w(self, code_):
        '''Append code to the the main list
        '''
        self.code.append((self.tab * self.level) + code_ + '\n')
        return
    
    def els(self, lines=1):
        '''Insert an empty line
        '''
        if (self.code[-1] != '\n') or (lines > 1):
            self.code.append(lines * '\n')
        return
    
    def stf(self, out_path):
        '''Save the list of code as a text file
        '''
        assert self.level == 0, \
            'Level should be zero instead of %d' % self.level
        
        with open(out_path, 'w') as out_hdl:
            out_hdl.writelines(self.code)
        return


def write_pyxbld(pyxbldcd):
    pyxbldcd.w('import os')
    pyxbldcd.w('from numpy import get_include')
    pyxbldcd.els()
    pyxbldcd.w('mod_dir = os.path.dirname(__file__)')
    pyxbldcd.els(2)
    pyxbldcd.w('def make_ext(modname, pyxfilename):')
    pyxbldcd.ind()
    pyxbldcd.w('from distutils.extension import Extension')
    pyxbldcd.w(('return Extension(name=modname, '
                        'sources=[pyxfilename], '
                        'language=\'c++\', '
                        'extra_compile_args=[r"/openmp", r"/Ox"], '
                        'include_dirs=[get_include(), mod_dir])'))
    pyxbldcd.ded(False)
    return

'''
@author: Faizan-Uni-Stuttgart

Template
'''
'''
import os
import timeit
import time
import subprocess
from pathlib import Path

from core import CodeGenr


def xxx(params_dict):
    module_name = 'xxx'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    infer_types = params_dict['infer_types']
    out_dir = params_dict['out_dir']
    
    pyxcd = CodeGenr(tab=tab)
    pxdcd = CodeGenr(tab=tab)
    pyxbldcd = CodeGenr(tab=tab)

    #==========================================================================
    # add cython flags
    #==========================================================================
    pyxcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pyxcd.w('# cython: boundscheck=%s' % boundscheck)
    pyxcd.w('# cython: wraparound=%s' % str(wraparound))
    pyxcd.w('# cython: cdivision=%s' % str(cdivision))
    pyxcd.w('# cython: language_level=%d' % int(language_level))
    pyxcd.w('# cython: infer_types(%s)' % str(infer_types))
    pyxcd.els()

    pxdcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pxdcd.w('# cython: boundscheck=%s' % boundscheck)
    pxdcd.w('# cython: wraparound=%s' % str(wraparound))
    pxdcd.w('# cython: cdivision=%s' % str(cdivision))
    pxdcd.w('# cython: language_level=%d' % int(language_level))
    pxdcd.w('# cython: infer_types(%s)' % str(infer_types))
    pxdcd.els()

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    pyxcd.w('from cython.parallel import (prange,  parallel)')
    pyxcd.els()

    pxdcd.w('import numpy as np')
    pxdcd.w('cimport numpy as np')
    pxdcd.els()

    #==========================================================================
    # declare types
    #==========================================================================
    pxdcd.w('ctypedef double DT_D')
    pxdcd.w('ctypedef unsigned long DT_UL')
    pxdcd.w('ctypedef long long DT_LL')
    pxdcd.w('ctypedef unsigned long long DT_ULL')
    pxdcd.w('ctypedef np.float64_t DT_D_NP_t')
    pxdcd.w('ctypedef np.uint64_t DT_UL_NP_t')
    pxdcd.els()

    pyxcd.w('DT_D_NP = np.float64')
    pyxcd.w('DT_UL_NP = np.uint64')
    pyxcd.els()

    pxdcd.w('DT_D_NP = np.float64')
    pxdcd.w('DT_UL_NP = np.uint64')
    pxdcd.els(2)   

    #==========================================================================
    # add external imports
    #==========================================================================

    #==========================================================================
    # Functions
    #==========================================================================

    #==========================================================================
    # w the pyxbld
    #==========================================================================
    pyxbldcd.w('import os')
    pyxbldcd.w('from numpy import get_include')
    pyxbldcd.els()
    pyxbldcd.w('mod_dir = os.path.dirname(__file__)')
    pyxbldcd.els(2)
    pyxbldcd.w('def make_ext(modname, pyxfilename):')
    pyxbldcd.ind()
    pyxbldcd.w('from distutils.extension import Extension')
    pyxbldcd.w(('return Extension(name=modname, '
                                 'sources=[pyxfilename], '
                                 'language=\'c++\', '
                                 'extra_compile_args=[r"/openmp"], '
                                 'include_dirs=[get_include(), mod_dir])'))
    pyxbldcd.ded(False)

    #==========================================================================
    # save as pyx, pxd, pyxbld
    #==========================================================================
    assert pyxcd.level == 0, \
        'Level should be zero instead of %d' % pyxcd.level
    assert pxdcd.level == 0, \
        'Level should be zero instead of %d' % pxdcd.level
    assert pyxbldcd.level == 0, \
        'Level should be zero instead of %d' % pyxbldcd.level
        
    out_path = os.path.join(out_dir, module_name)
    pyxcd.stf(out_path + '.pyx')
    pxdcd.stf(out_path + '.pxd')
    pyxbldcd.stf(out_path + '.pyxbld')

#     #==========================================================================
#     # Check for syntax errors
#     #==========================================================================
#     abs_path = os.path.abspath(out_path + '.pyx')
#     arg = (cython, "%s -a" % abs_path)
#     subprocess.call([arg])

    return
    

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    
    main_dir = Path(os.getcwd())
    
    tab = '    '
    nonecheck = False
    boundscheck = False
    wraparound = False
    cdivision = True
    language_level = 3
    infer_types = None
    out_dir = os.getcwd()
    
    os.chdir(main_dir)
    
    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir 
    xxx(params_dict)
    
    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))

'''

