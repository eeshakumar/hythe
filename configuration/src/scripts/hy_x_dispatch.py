from argparse import ArgumentParser
import sys
import os
from hythe.libs.dispatch.remote_dispatcher import RemoteDispatcher


def get_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--target", type=str, default="hy-x-run",
                        help="Enter the target build to deploy and execute.")
    parser.add_argument("--username", type=str, default="ekumar",
                        help="Enter the username for authentication. "
                             "The authentication will be completed with registered SHA-256 private key.")
    parser.add_argument("--remotepath", type=str, default=None,
                        help="Enter the path to remote home.")
    parser.add_argument("--remoteroot", type=str, default="auto_default",
                        help="Enter the remote ws root to save experiment to.")
    parser.add_argument("--bazelwsroot", type=str, default=None,
                        help="Enter the local bazel ws root.")
    parser.add_argument("--hostname", type=str, default="192.168.16.89",
                        help="Enter the remote hostname.")
    parser.add_argument("--jobname", type=str, default=None,
                        help="Enter the slurm jobname.")
    return parser


def main():
    private_key_file = "~/.ssh/id_{}_cluster"
    parsed_args = get_args().parse_args(sys.argv[1:])
    private_key_file = private_key_file.format(parsed_args.username)
    dispatcher = RemoteDispatcher(hostname=parsed_args.hostname,
                                  username=parsed_args.username,
                                  private_key_file=private_key_file)
    if dispatcher.connect():
        print("Authentication successful!")
        #
        # bazel_bin_root = os.path.join(parsed_args.bazelwsroot, "bazel-bin/configuration")
        # dispatcher.deploy(bazel_bin_root, parsed_args.remotepath)
        cmd = "sbatch --job-name={jobname} run.sh {target} --jobname={jobname}".format(
            target=parsed_args.target, jobname=parsed_args.jobname)
        print()
        dispatcher.execute(cmd, exec_path=os.path.join(
            parsed_args.remotepath, parsed_args.remoteroot))
        dispatcher.close()
    return


if __name__ == "__main__":
    main()