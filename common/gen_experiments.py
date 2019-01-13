from itertools import product
import os
import json
from subprocess import Popen
import logging


def find_variables(param_dict):
    """Find items in the dictionnary that are lists and consider them as variables."""
    variables = []
    for key, val in param_dict.items():
        if isinstance(val, list):
            variables.append(key)
    return variables


def grid_search(param_dict):
    """Find the variables in param_dict and yields every instance part of the cartesian product.
    
    args:
        param_dict: dictionnary of parameters. Every item that is a list will be crossvalidated.
    
    yields: A dictionnary of parameters where lists are replaced with one of their instance.
    """
    variables = []
    for key, val in param_dict.items():
        if isinstance(val, list):
            variables.append([(key, element) for element in val])

    for experiment in product(*variables):
        yield dict(experiment)


def make_experiment_name(experiment):
    """Create a readable name containing the name and value of the variables."""
    args = []
    for name, value in experiment.items():
        if isinstance(value, float):
            args.append("%s=%.4g" % (name, value))
        else:
            args.append("%s=%s" % (name, value))
    return ';'.join(args)


def gen_experiments_dir(param_dict, root_dir, exp_description, cmd=None, blocking=False, borgy_args=None):
    """Generate all directories with their json and launch cmd with the flag --exp_dir."""
    process_list = []
    for i, experiment in enumerate(grid_search(param_dict)):
        name = make_experiment_name(experiment)
        print("Exp %d: %s." % (i, name))
        param_dict.update(experiment)

        exp_dir_borgy = os.path.join(root_dir, name)
        exp_dir = '/mnt/' + exp_dir_borgy
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        param_path = os.path.join(exp_dir, 'params.json')

        with open(param_path, 'w') as fd:
            json.dump(param_dict, fd, indent=4)

        if cmd is not None:

            if borgy_args:
                cmd_ = """cd '%s'; stdbuf -oL '%s' --exp_dir='%s' 1>>stdout 2>>stderr""" %(exp_dir, cmd, exp_dir)
                args = ['borgy', 'submit', '--name', "%s_(%s)" % (exp_description, name)] + borgy_args + ['--', 'bash', '-c', cmd_, ]
                str_cmd = ' '.join(['"'+arg+'"' for arg in args])
                print(str_cmd)
                
                with open(os.path.join(exp_dir, 'borgy_submit.cmd'), 'w') as fd:
                    fd.write(str_cmd)
                
                process = Popen(args)
                if blocking:
                    process.wait()
                process_list.append(process)

            else:
                args = [cmd, '--exp_dir=%s' % exp_dir]
                with open(os.path.join(exp_dir, 'stderr'), 'w') as err_fd:
                    with open(os.path.join(exp_dir, 'stdout'), 'w') as out_fd:
                        process_list.append(Popen(args, stderr=err_fd, stdout=out_fd))

    if blocking:
        for process in process_list:
            process.wait()


def re_run(root_dir, cmd=None, blocking=False, borgy_args=None, exp_dir_list=None):
    process_list = []

    for exp in exp_dir_list or os.listdir(root_dir):
        exp_dir = os.path.join(root_dir, exp)
        # Remove the "mnt" part
        exp_dir = "/" + "/".join((exp_dir.split("/")[2:]))

        if cmd is not None:

                if borgy_args:
                    cmd_ = """cd '%s'; stdbuf -oL '%s' --exp_dir='%s' 1>>stdout 2>>stderr""" %(exp_dir, cmd, exp_dir)
                    args = ['borgy', 'submit', '--name', "%s" % exp] + borgy_args + ['--', 'bash', '-c', cmd_, ]
                    print(' '.join(args))
                    process_list.append(Popen(args))

                else:
                    args = [cmd, '--exp_dir=%s' % exp_dir]
                    with open(os.path.join(exp_dir, 'stderr'), 'w') as err_fd:
                        with open(os.path.join(exp_dir, 'stdout'), 'w') as out_fd:
                            process_list.append(Popen(args, stderr=err_fd, stdout=out_fd))

        if blocking:
            for process in process_list:
                process.wait()


def load_and_save_params(default_params, exp_dir, ignore_existing=False):
    """Update default_params with params.json from exp_dir and overwrite params.json with updated version."""
    default_params = json.loads(json.dumps(default_params))
    param_path = os.path.join(exp_dir, 'params.json')
    logging.info("Searching for '%s'" % param_path)
    if os.path.exists(param_path) and not ignore_existing:
        logging.info("Loading existing params.")
        with open(param_path, 'r') as fd:
            params = json.load(fd)
        default_params.update(params)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    with open(param_path, 'w') as fd:
        json.dump(default_params, fd, indent=4)

    return default_params
