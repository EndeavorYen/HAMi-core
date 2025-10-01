.DEFAULT_GOAL := build

current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

build:
	sh ./build.sh
.PHONY: build

build-in-docker:
	docker run -i --rm \
		-v $(current_dir):/libvgpu \
		-w /libvgpu \
		-e DEBIAN_FRONTEND=noninteractive \
		nvidia/cuda:12.2.0-devel-ubuntu20.04 \
		sh -c "sed -i 's|archive.ubuntu.com|tw.archive.ubuntu.com|g' /etc/apt/sources.list; apt-get -y update; apt-get -y install cmake; bash ./build.sh"
.PHONY: build-in-docker
