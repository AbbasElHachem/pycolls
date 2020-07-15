'''
Created on Dec 7, 2017

@author: Faizan-Uni
'''

# import os
# import timeit
# import time
from pathlib import Path
from fnmatch import fnmatch

from .write_cp_classi_main import write_cp_classi_main_lines
from .write_gen_mod_cp_rules import write_gen_mod_cp_rules_lines
from .write_memb_ftns import write_memb_ftns_lines
from .write_obj_ftns import write_obj_ftns_lines
from .write_qual_objs import write_qual_obj_lines


def create_classi_cython_files(obj_1_flag=False,
                               obj_2_flag=False,
                               obj_3_flag=False,
                               obj_4_flag=False,
                               obj_5_flag=False,
                               obj_6_flag=False,
                               obj_7_flag=False,
                               obj_8_flag=False,
                               nonecheck=True,
                               boundscheck=True,
                               wraparound=True,
                               cdivision=True,
                               infer_types=None,
                               language_level=3,
                               force_compile=False,
                               out_dir='',
                               op_mp_memb_flag=True,
                               op_mp_obj_ftn_flag=True):

    assert any([obj_1_flag,
                obj_2_flag,
                obj_3_flag,
                obj_4_flag,
                obj_5_flag,
                obj_6_flag,
                obj_7_flag,
                obj_8_flag])
    assert out_dir

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    params_dict['obj_1_flag'] = obj_1_flag
    params_dict['obj_2_flag'] = obj_2_flag
    params_dict['obj_3_flag'] = obj_3_flag
    params_dict['obj_4_flag'] = obj_4_flag
    params_dict['obj_5_flag'] = obj_5_flag
    params_dict['obj_6_flag'] = obj_6_flag
    params_dict['obj_7_flag'] = obj_7_flag
    params_dict['obj_8_flag'] = obj_8_flag

    params_dict['op_mp_memb_flag'] = op_mp_memb_flag
    params_dict['op_mp_obj_ftn_flag'] = op_mp_obj_ftn_flag
    
    out_dir = Path(out_dir)
    
    path_to_main_pyx = out_dir / 'cp_classi_main.pyx'
    path_to_cp_rules_pyx = out_dir / 'gen_mod_cp_rules.pyx'
    path_to_memb_ftns_pyx = out_dir / 'memb_ftns.pyx'

    compile_classi_main = False
    compile_gen_mod_cp_rules = False
    compile_memb_ftns = True

    if path_to_main_pyx.exists() and (not force_compile):
        mtch_str = ''
        compile_classi_main = True
        new_flags_list = [obj_1_flag,
                          obj_2_flag,
                          obj_3_flag,
                          obj_4_flag,
                          obj_5_flag,
                          obj_6_flag,
                          obj_7_flag,
                          obj_8_flag]
        
        with open(path_to_main_pyx, 'r') as pyx_hdl:
            # obj flags
            for line in pyx_hdl:
                if fnmatch(line, '### obj_ftns:*'):
                    mtch_str = line
                    break

            if mtch_str:
                _ = (mtch_str.split(':')[1]).strip().split(';')
                old_flags_list = [True if x == 'True' else False for x in _]

                if new_flags_list == old_flags_list:
                    compile_classi_main = False

            if not compile_classi_main:
                # open mp obj flag
                for line in pyx_hdl:
                    if fnmatch(line, '### op_mp_obj_ftn_flag:*'):
                        mtch_str = line
                        break

                if mtch_str:
                    _ = (mtch_str.split(':')[1]).strip()
                    if _ != str(op_mp_obj_ftn_flag):
                        compile_classi_main = True

    else:
        compile_classi_main = True

    if (not path_to_cp_rules_pyx.exists()) or force_compile:
        compile_gen_mod_cp_rules = True

    if force_compile  or (not path_to_memb_ftns_pyx.exists()):
        pass
    else:
        if path_to_memb_ftns_pyx.exists():
            # open mp membership flag
            with open(path_to_memb_ftns_pyx, 'r') as pyx_hdl:
                mtch_str = ''
                for line in pyx_hdl:
                    if fnmatch(line, '### op_mp_memb_flag:*'):
                        mtch_str = line
                        break

                if mtch_str:
                    _ = (mtch_str.split(':')[1]).strip()
                    if _ == str(op_mp_memb_flag):
                        compile_memb_ftns = False

    if compile_classi_main:
        write_cp_classi_main_lines(params_dict)
        write_obj_ftns_lines(params_dict)

    if compile_gen_mod_cp_rules:
        write_gen_mod_cp_rules_lines(params_dict)

    if compile_memb_ftns:
        write_memb_ftns_lines(params_dict)

    return


def create_justi_cython_files(nonecheck=True,
                              boundscheck=True,
                              wraparound=True,
                              cdivision=True,
                              infer_types=None,
                              language_level=3,
                              force_compile=False,
                              out_dir='',
                              op_mp_memb_flag=True):

    assert out_dir

    out_dir = Path(out_dir)
    path_to_memb_ftns_pyx = out_dir / 'memb_ftns.pyx'

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir
    params_dict['op_mp_memb_flag'] = op_mp_memb_flag
    
    compile_memb_ftns = True

    if force_compile  or (not path_to_memb_ftns_pyx.exists()):
        pass
    else:
        if path_to_memb_ftns_pyx.exists():
            # open mp membership flag
            with open(path_to_memb_ftns_pyx, 'r') as pyx_hdl:
                mtch_str = ''
                for line in pyx_hdl:
                    if fnmatch(line, '### op_mp_memb_flag:*'):
                        mtch_str = line
                        break

                if mtch_str:
                    _ = (mtch_str.split(':')[1]).strip()
                    if _ == str(op_mp_memb_flag):
                        compile_memb_ftns = False

    if compile_memb_ftns:
        write_memb_ftns_lines(params_dict)

    return


def create_obj_cython_files(obj_1_flag=False,
                            obj_2_flag=False,
                            obj_3_flag=False,
                            obj_4_flag=False,
                            obj_5_flag=False,
                            obj_6_flag=False,
                            obj_7_flag=False,
                            obj_8_flag=False,
                            nonecheck=True,
                            boundscheck=True,
                            wraparound=True,
                            cdivision=True,
                            infer_types=None,
                            language_level=3,
                            force_compile=False,
                            out_dir='',
                            op_mp_memb_flag=True,
                            op_mp_obj_ftn_flag=True):

    assert any([obj_1_flag,
                obj_2_flag,
                obj_3_flag,
                obj_4_flag,
                obj_5_flag,
                obj_6_flag,
                obj_7_flag,
                obj_8_flag])
    assert out_dir

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    params_dict['obj_1_flag'] = obj_1_flag
    params_dict['obj_2_flag'] = obj_2_flag
    params_dict['obj_3_flag'] = obj_3_flag
    params_dict['obj_4_flag'] = obj_4_flag
    params_dict['obj_5_flag'] = obj_5_flag
    params_dict['obj_6_flag'] = obj_6_flag
    params_dict['obj_7_flag'] = obj_7_flag
    params_dict['obj_8_flag'] = obj_8_flag

    params_dict['op_mp_obj_ftn_flag'] = op_mp_obj_ftn_flag

    out_dir = Path(out_dir)

    path_to_obj_main_pyx = out_dir / 'obj_vals_main.pyx'
    path_to_obj_pyx = out_dir / 'cp_obj_ftns.pyx'

    compile_classi_main = True

    if (path_to_obj_pyx.exists() and
        (not force_compile)):
        mtch_str = ''
        new_flags_list = [obj_1_flag,
                          obj_2_flag,
                          obj_3_flag,
                          obj_4_flag,
                          obj_5_flag,
                          obj_6_flag,
                          obj_7_flag,
                          obj_8_flag]

        with open(path_to_obj_pyx, 'r') as pyx_hdl:
            # obj flags
            for line in pyx_hdl:
                if fnmatch(line, '### obj_ftns:*'):
                    mtch_str = line
                    break

            if mtch_str:
                _ = (mtch_str.split(':')[1]).strip().split(';')
                old_flags_list = [True if x == 'True' else False for x in _]

                if new_flags_list == old_flags_list:
                    compile_classi_main = False

            if not compile_classi_main:
                # open mp obj flag
                for line in pyx_hdl:
                    if fnmatch(line, '### op_mp_obj_ftn_flag:*'):
                        mtch_str = line
                        break

                if mtch_str:
                    _ = (mtch_str.split(':')[1]).strip()
                    if _ != str(op_mp_obj_ftn_flag):
                        compile_classi_main = True

    if (path_to_obj_main_pyx.exists() and
        (not force_compile) and
        (not compile_classi_main)):
        mtch_str = ''
        compile_classi_main = True
        new_flags_list = [obj_1_flag,
                          obj_2_flag,
                          obj_3_flag,
                          obj_4_flag,
                          obj_5_flag,
                          obj_6_flag,
                          obj_7_flag,
                          obj_8_flag]

        with open(path_to_obj_main_pyx, 'r') as pyx_hdl:
            # obj flags
            for line in pyx_hdl:
                if fnmatch(line, '### obj_ftns:*'):
                    mtch_str = line
                    break

            if mtch_str:
                _ = (mtch_str.split(':')[1]).strip().split(';')
                old_flags_list = [True if x == 'True' else False for x in _]

                if new_flags_list == old_flags_list:
                    compile_classi_main = False

            if not compile_classi_main:
                # open mp obj flag
                for line in pyx_hdl:
                    if fnmatch(line, '### op_mp_obj_ftn_flag:*'):
                        mtch_str = line
                        break

                if mtch_str:
                    _ = (mtch_str.split(':')[1]).strip()
                    if _ != str(op_mp_obj_ftn_flag):
                        compile_classi_main = True

    if compile_classi_main:
        write_qual_obj_lines(params_dict)
        write_obj_ftns_lines(params_dict)
    return


def gen_mod_cp_rules_cyth_files(nonecheck=True,
                                boundscheck=True,
                                wraparound=True,
                                cdivision=True,
                                infer_types=None,
                                language_level=3,
                                force_compile=False,
                                out_dir=''):
 
    assert out_dir

    tab = '    '

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir
    
    out_dir = Path(out_dir)
    path_to_gen_mod_cp_rules_pyx = out_dir / 'gen_mod_cp_rules.pyx'
    
    if (not path_to_gen_mod_cp_rules_pyx.exists()) or force_compile:
        compile_gen_mod_cp_rules_ftns = True
    else:
        compile_gen_mod_cp_rules_ftns = False
    
    if compile_gen_mod_cp_rules_ftns:
        write_gen_mod_cp_rules_lines(params_dict)

    return
