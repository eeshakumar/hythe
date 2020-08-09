import paramiko
from os import path
from hythe.libs.dispatch.dispatcher import Dispatcher


class RemoteDispatcher(Dispatcher):

    def __init__(self, username, dispatch_dict=None, hostname="192.168.16.89",
                 private_key_file="~/.ssh/id_ekumar_cluster",
                 target="hy-x-run"):
        super(RemoteDispatcher, self).__init__(dispatch_dict)
        self.private_key_file = path.expanduser(private_key_file)
        self._hostname = hostname
        self._username = username
        self._target = target
        self._pkey = paramiko.RSAKey.from_private_key_file(self.private_key_file)
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def is_connect(self):
        return

    def dispatch(self):
        return

    def connect(self):
        conn = self._ssh_client.connect(hostname=self._hostname, username=self._username,
                                pkey=self._pkey)
        self._sftp = self._ssh_client.open_sftp()
        return conn is None

    def deploy(self, local_path, remote_path, remote_root="auto_default"):
        return


    def execute(self, cmd, exec_path=None):
        if exec_path is not None:
            stdin, stdout, stderr = self._ssh_client.exec_command("cd {};".format(exec_path) + cmd)
        else:
            stdin, stdout, stderr = self._ssh_client.exec_command(cmd)
        print(stdout.readlines())
        return

    def close(self):
        self._sftp.close()
        self._ssh_client.close()

    @staticmethod
    def uploading_info(uploaded_file_size, total_file_size):
        print('uploaded_file_size : {} total_file_size : {}'.
              format(uploaded_file_size, total_file_size))