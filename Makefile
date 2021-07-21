.PHONY: help
help:
	@echo "make: The okay-est command line tool"
	@echo
	@echo "usage: make <command> [CMD_ARGS='command args']"
	@echo
	@echo "<command> is one of:"
	@echo
	@printf "%16s: %s\n" "build" "Build docker image"
	@printf "%16s: %s\n" "clean" "Deletes all temporary files"
	@printf "%16s: %s\n" "shell" "Run docker in interactive mode"
	@echo
	@echo "Additional arguments:"
	@echo
	@printf "\t%24s %s\n" "- CPU_ONLY to run docker without GPU support"
	@echo
	@echo "Examples:"
	@echo
	@echo "        Default shell mode (ok for GCP machine with GPU)"
	@echo "          make shell"
	@echo
	@echo "        Shell mode CPU only :"
	@echo "          make shell CPU_ONLY=1"
	@echo

image_name = fairmot
shell=/bin/bash
enable_gpu = --gpus all
docker_run_args = --ipc=host
volumes = -v ${HOME}/.netrc:/root/.netrc \
          -v ${HOME}/.bash_history:/root/.bash_history \
		  -v ${HOME}/data:/app/data \
		  -v $(shell pwd):/app/FairMOT
user_id = $(shell id -u)
group_id = $(shell id -g)
user_group = ${user_id}:${group_id}
run_as_user = --user ${user_group}
workdir = -w /app/FairMOT

ifeq (${CPU_ONLY}, 1)
$(info CPU ONLY)
enable_gpu =
endif

.PHONY: build
build:
	@docker build --target local -t ${image_name} .

.PHONY: clean
clean:
	@find . -name "*.pyc" -exec rm -f {} \;

.PHONY: shell
shell:
	@docker run ${enable_gpu} \
	--rm -it ${docker_run_args} ${volumes} \
	${run_as_user} ${workdir} ${image_name}
