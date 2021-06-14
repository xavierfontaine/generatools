import unittest
import copy
import mlflow
import tempfile
import shutil
import generatools.utils.mlflow


class MlflowTester(unittest.TestCase):
    """Tester with setUp and tearDown creating an expe folder and deleting
    it"""

    def setUp(self):
        self.expes_uri = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.expes_uri)


class TestMetricExist(MlflowTester):
    """
    Test type: private (implementation detail)
    """

    def test_two_cases(self):
        """When metric already exists and when does not"""
        # Variables
        expe_name = "temp_expe"
        run_id = "0"
        metric_name = "temp_metric"
        #  Core
        mlflow.set_tracking_uri(uri=self.expes_uri)
        experiment_id = mlflow.create_experiment(name=expe_name)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            self.assertFalse(
                generatools.utils.mlflow.check_metrics_exist(
                    tracking_uri=self.expes_uri, run_id=run_id, key=metric_name
                )
            )
            mlflow.log_metric(key=metric_name, value=10)
            self.assertTrue(
                generatools.utils.mlflow.check_metrics_exist(
                    tracking_uri=self.expes_uri, run_id=run_id, key=metric_name
                )
            )


class TestDictToMlflowParams(unittest.TestCase):
    def test_depth_1_dic(self):
        dic = {"k1": "v1", "k2": "v2"}
        exp_out = copy.deepcopy(dic)
        obs_out = generatools.utils.mlflow.dict_to_mlflow_params(dic=dic)
        self.assertEqual(exp_out, obs_out)

    def test_depth_2_dic(self):
        dic = {"k1": "v1", "k2": {"k21": "v21", "k22": "v22"}}
        exp_out = {"k1": "v1", "k2__k21": "v21", "k2__k22": "v22"}
        obs_out = generatools.utils.mlflow.dict_to_mlflow_params(dic=dic)
        self.assertEqual(exp_out, obs_out)

    def test_transf_func(self):
        def foo(x):
            return x

        dic = {"k1": "v1", "k2": {"k21": foo}}
        exp_out = {
            "k1": "v1",
            "k2__k21": "        def foo(x):\n            return x\n",
        }
        obs_out = generatools.utils.mlflow.dict_to_mlflow_params(dic=dic)
        self.assertEqual(exp_out, obs_out)
