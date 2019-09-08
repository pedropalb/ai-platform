import json
import shutil

import click
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.telemetry import set_diagnostics_collection
from azureml.tensorboard import Tensorboard
from azureml.train.dnn import TensorFlow


@click.command()
@click.option('--local-data-path', required=False, help="The path to the data which will be uploaded.")
def azure_train(local_data_path):
    set_diagnostics_collection(send_diagnostics=True)
    ws = Workspace.from_config()

    exp = Experiment(workspace=ws, name='orientation-correction')

    ds = ws.get_default_datastore()

    if local_data_path is not None:
        ds.upload(src_dir=local_data_path,
                  target_path='data',
                  overwrite=True,
                  show_progress=True)

    # choose a name for your cluster
    cluster_name = "gpu-cluster"

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                               max_nodes=1)
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    script_folder = './azure_scripts'
    shutil.copy('./train.py', script_folder)
    shutil.copy('./train_config.json', script_folder)
    shutil.copy('./dataset.py', script_folder)
    shutil.copy('./model.py', script_folder)
    shutil.copy('./callbacks.py', script_folder)

    script_params = {
        '--data-path': ds.path('data').as_download(),
        '--config-path': './train_config.json'
    }

    est = TensorFlow(source_directory=script_folder,
                     script_params=script_params,
                     compute_target=compute_target,
                     entry_script='train.py',
                     pip_packages=['keras==2.2.5', 'pillow==6.1.0'],
                     use_gpu=True,
                     framework_version='1.13')


    run = exp.submit(est)
    tb = Tensorboard([run], local_root='./train_logs')
    tb.start()
    print(run.get_portal_url())
    run.wait_for_completion(show_output=True)
    tb.stop()


if __name__ == '__main__':
    azure_train()
