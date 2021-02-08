docker build -t pv_and_malfunction_detector .
docker run --interactive --tty  pv_and_malfunction_detector
docker run -v C:/Users/FellnerD/Desktop:/usr/src/app --interactive --tty  pv_and_malfunction_detector

make sure to have a docker running, run docker in interactive mode (--interactive) and have the drive with your data accessible to docker!
classic fixes for docker problems: run docker and powershell in admin mode; if porblems with daemon run : cd "C:\Program Files\Docker\Docker"
										       ./DockerCli.exe -SwitchDaemon