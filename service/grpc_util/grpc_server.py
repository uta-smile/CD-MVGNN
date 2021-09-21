import os
import grpc
import time
import copy
from concurrent import futures
from service.moleprop import MoleProp
from dglt.utils import create_logger
from mpp_service import get_available_tasks
from service.grpc_util.proto import message_pb2, message_pb2_grpc
from multiprocessing import Pool, Process, Queue
#from torch.multiprocessing import Pool, Process
import threading
from threading import Thread
import traceback

debug = create_logger(name='server', save_dir='service/log', quiet=False).debug

def predict_task_thread(task, mps, input_queue, output_queue):
    for smiles in iter(input_queue.get, 'STOP'):
        try:
            output = mps.predict(task=task, smiles=smiles)
            while not output_queue.empty():
                output_queue.get()
            output_queue.put(output)
        except Exception as e:
            output_queue.put({})
            debug(traceback.format_exc())

def predict_task_process(tasks, task_argss, checkpoint_dir, input_queues, output_queues, input_queues_file, output_queues_file):
    assert len(tasks) == len(task_argss) and \
           len(tasks) == len(input_queues) and \
           len(tasks) == len(output_queues) and \
           len(tasks) == len(input_queues_file) and \
           len(tasks) == len(output_queues_file)
    ts, ts_file = [], []
    for i in range(len(tasks)):
        mp = MoleProp(checkpoint_dir=os.path.join(checkpoint_dir, tasks[i]), debug=print)
        mp.load_model(task_argss[i])
        t = Thread(target=predict_task_thread, args=(tasks[i], mp, input_queues[i], output_queues[i],))
        t_file = Thread(target=predict_task_thread, args=(tasks[i], mp, input_queues_file[i], output_queues_file[i],))
        ts.append(t)
        ts_file.append(t_file)

    for i in range(len(ts)):
        ts[i].start()
        ts_file[i].start()

    for i in range(len(ts)):
        ts[i].join()
        ts_file[i].join()

class Server(message_pb2_grpc.MolePropServerServicer):
    """Server of Molecular Properties Prediction"""
    def __init__(self, args, tasks, checkpoint_dir, gpu=True, group_size = 2, max_workers=1, ip='localhost', port='8888'):
        if not os.path.exists(checkpoint_dir):
            raise Exception('checkpoint_dir: "{}" not found'.format(checkpoint_dir))
        self.ip_ = ip
        self.port_ = port

        debug('\n\nss{} args:{}\ntasks:{}\ncheckpoint_dir:{}\nmax_workers:{}\nip:{}\nport:{}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            args,
            tasks,
            checkpoint_dir,
            max_workers,
            ip,
            port
        ))
        mps = []
        input_queue = {}
        output_queue = {}
        mutex = threading.Lock()
        input_file_queue = {}
        output_file_queue = {}
        mutex_file = threading.Lock()

        args.cuda = gpu
        task_g, task_argss, input_queues, output_queues, input_queues_file, output_queues_file = [], [], [], [], [], []
        for i, task in enumerate(tasks):
            task_args = copy.deepcopy(args)
            mps.append(task)
            input_queue[task] = Queue()
            output_queue[task] = Queue()
            input_file_queue[task] = Queue()
            output_file_queue[task] = Queue()
            task_g.append(task)
            task_argss.append(task_args)
            input_queues.append(input_queue[task])
            output_queues.append(output_queue[task])
            input_queues_file.append(input_file_queue[task])
            output_queues_file.append(output_file_queue[task])
            if len(task_g) == group_size or i+1 == len(tasks):
                Process(target=predict_task_process, args=(task_g, task_argss, checkpoint_dir, input_queues, output_queues, input_queues_file, output_queues_file,)).start()
                task_g, task_argss, input_queues, output_queues, input_queues_file, output_queues_file = [], [], [], [], [], []

        debug('{} {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Load models done!'))

        class Servicer(message_pb2_grpc.MolePropServerServicer):
            def __init__(self):
                self.request_num_ = 0

            def get_mole_prop(self, request, context):
                """
                Get molecular properties.
                :param request: request data.
                :param context:
                :return:
                """
                self.request_num_ += 1
                tasks = list(set(request.tasks))
                debug('\n\n\n{} request_num:{}\nclient:{}\ntasks:{}\nsmiles_len:{}\nsmiles:{}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    self.request_num_,
                    context.peer(),
                    tasks,
                    len(request.smiles),
                    request.smiles
                ))
                for task in tasks:
                    if task not in mps:
                        msg = 'Sorry, {} is not a valid task, task can be {}'.format(task, get_available_tasks())
                        debug('{} msg:{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg))
                        return message_pb2.Response(msg=msg, score=None)

                score = {}
                tic = time.time()
                if len(request.smiles) == 1:
                    in_queue = input_queue
                    out_queue = output_queue
                    lock = mutex
                else:
                    in_queue = input_file_queue
                    out_queue = output_file_queue
                    lock = mutex_file
                with lock:
                    for i, task in enumerate(tasks):
                        in_queue[task].put(list(request.smiles))
                        debug('{} i:{} task:{} put'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, task))
                    for i, task in enumerate(tasks):
                        task_scores = out_queue[task].get()
                        if len(task_scores) > 0:
                            debug('{} i:{} task:{} scores_len:{} get\n{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i+1, task, len(task_scores['task_score']), task_scores))
                            for key in task_scores['task_score']:
                                task_scores['task_score'][key] = message_pb2.ListValue(val=task_scores['task_score'][key])
                            score[task_scores['task']] = message_pb2.TaskScore(task_score=task_scores['task_score'])
                        else:
                            debug('{} i:{} task:{} get NULL'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, task))
                toc = time.time()

                # tic = time.time()
                # if len(request.smiles) == 1:
                #     lock = threading.Lock()
                #     with lock:
                #         score = self.get_single_result(request)
                # else:
                #     score = self.get_multiple_results(request)
                # toc = time.time()

                # score = {}
                # tic = time.time()
                # for i, task in enumerate(tasks):
                #     input_queue[task].put(list(request.smiles))
                #     debug('{} i:{} task:{} put'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, task))
                # for i, task in enumerate(tasks):
                #     task_scores = output_queue[task].get()
                #     if len(task_scores) > 0:
                #         debug('{} i:{} task:{} scores_len:{} get\n{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i+1, task, len(task_scores['task_score']), task_scores))
                #         for key in task_scores['task_score']:
                #             task_scores['task_score'][key] = message_pb2.ListValue(val=task_scores['task_score'][key])
                #         score[task_scores['task']] = message_pb2.TaskScore(task_score=task_scores['task_score'])
                #     else:
                #         debug('{} i:{} task:{} get NULL'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i + 1, task))
                # toc = time.time()

                if len(score) > 0:
                    msg = 'success'
                else:
                    msg = 'failed'

                debug('{} cost_time:{}s\nmsg:{}'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    (toc - tic),
                    msg
                ))

                return message_pb2.Response(msg=msg, score=score)
        self.server_ = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        message_pb2_grpc.add_MolePropServerServicer_to_server(Servicer(), self.server_)

    def start(self):
        """
        Start server
        :return:
        """
        self.server_.add_insecure_port('{}:{}'.format(self.ip_, self.port_))
        self.server_.start()
        try:
            while True:
                time.sleep(60*60*24)
        except KeyboardInterrupt:
            self.server_.stop(0)