import os
from typing import Dict

import attr
import numpy as np

import rlf.rl.utils as rutils
import rlf
import pandas as pd


@attr.s(auto_attribs=True, slots=True)
class RunResult:
    prefix: str
    eval_result: Dict = {}


def run_policy(run_settings, runner=None):
    if runner is None:
        print("runner is None, runner = run_settings.create_runner()")
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args
    if args.__contains__('num_epochs'):
        sudo_batch_size = end_update // args.num_epochs
    if args.__contains__('bc_num_epochs'):
        sudo_batch_size = end_update // args.bc_num_epochs

    if args.ray:
        import ray
        from ray import tune

        # Release resources as they will be recreated by Ray
        runner.close()

        use_config = eval(args.ray_config)
        use_config["cwd"] = os.getcwd()
        use_config = run_settings.get_add_ray_config(use_config)

        rutils.pstart_sep()
        print("Running ray for %i updates per run" % end_update)
        rutils.pend_sep()

        ray.init(local_mode=args.ray_debug)
        tune.run(
            type(run_settings),
            resources_per_trial={"cpu": args.ray_cpus, "gpu": args.ray_gpus},
            stop={"training_iteration": end_update},
            num_samples=args.ray_nsamples,
            global_checkpoint_period=np.inf,
            config=use_config,
            **run_settings.get_add_ray_kwargs()
        )
    else:
        args = runner.args

        if runner.should_load_from_checkpoint():
            runner.load_from_checkpoint()

        if args.eval_only:
            eval_result = runner.full_eval(run_settings.create_traj_saver)
            return RunResult(prefix=args.prefix, eval_result=eval_result)

        start_update = 0
        if args.resume:
            start_update = runner.resume()

        runner.setup()
        print("RL Training (%d/%d)" % (start_update, end_update))

        if runner.should_start_with_eval:
            runner.eval(-1)

        train_info_list = []
        eval_info_list = []
        # Initialize outside the loop just in case there are no updates.
        j = 0
        print("start_update:", start_update)
        print("end_update:", end_update)
        print("runner.updater.split", str(type(runner.updater)).split('.')[-1].split('\'')[0])
        for j in range(start_update, end_update):
            updater_log_vals = runner.training_iter(j)
            if args.log_interval > 0 and (j + 1) % args.log_interval == 0:
                log_dict = runner.log_vals(updater_log_vals, j)
            if (j+1) % args.save_interval == 0:
                runner.save(j)
            if (j+1) % args.eval_interval == 0:
                goal_achieved = runner.eval(j)
        
        if args.eval_interval > 0:
            runner.eval(j + 1)
        if args.save_interval > 0:
            runner.save(j + 1)
        runner.close()
        # WB prefix of the run so we can later fetch the data.
        return RunResult(args.prefix)


def evaluate_policy(run_settings, runner=None):
    if runner is None:
        runner = run_settings.create_runner()
    end_update = runner.updater.get_num_updates()
    args = runner.args

    if args.ray:
        import ray
        from ray import tune

        # Release resources as they will be recreated by Ray
        runner.close()

        use_config = eval(args.ray_config)
        use_config["cwd"] = os.getcwd()
        use_config = run_settings.get_add_ray_config(use_config)

        rutils.pstart_sep()
        print("Running ray for %i updates per run" % end_update)
        rutils.pend_sep()

        ray.init(local_mode=args.ray_debug)
        tune.run(
            type(run_settings),
            resources_per_trial={"cpu": args.ray_cpus, "gpu": args.ray_gpus},
            stop={"training_iteration": end_update},
            num_samples=args.ray_nsamples,
            global_checkpoint_period=np.inf,
            config=use_config,
            **run_settings.get_add_ray_kwargs()
        )
    else:
        args = runner.args
        
        print("")
        print("runner.should_load_from_checkpoint():", runner.should_load_from_checkpoint())
        print("args.eval_only:", args.eval_only)
        print("args.resume:", args.resume)
        print("")
        
        if runner.should_load_from_checkpoint():
            runner.load_from_checkpoint()
 
        if args.eval_only:
            eval_result, goal_achieved, eval_step_list= runner.full_eval(run_settings.create_traj_saver)
            succ_steps = [b for a, b in zip( goal_achieved, eval_step_list) if a]
            print("ave_succ_steps:", sum(succ_steps) / len(succ_steps))

            epoch = list(range(1, len(goal_achieved)+2))
            succ_rate = np.sum(goal_achieved) / args.num_eval
            goal_achieved.append(succ_rate)
            eval_step_list.append(0)
            dataframe = pd.DataFrame({'epoch': epoch, 'goal_achieved':goal_achieved, 'eval_step_list:': eval_step_list})
            algo = str(type(runner.updater)).split('.')[-1].split('\'')[0]
            dataframe.to_csv(algo + "_test.csv", index=False, sep=',')
            print("succ_rate:", succ_rate)
            return RunResult(prefix=args.prefix, eval_result=eval_result)

        start_update = 0
        if args.resume:
            start_update = runner.resume()

        # WB prefix of the run so we can later fetch the data.
        return RunResult(args.prefix)
