FROM ubuntu:14.04

MAINTAINER Byron Tasseff <byron@tasseff.com>

ENV HOME /root
WORKDIR $HOME/

ADD . nuflood/

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install gcc g++ make cmake

RUN cd /root/nuflood/ && mkdir build && cd build/ && cmake .. && make
RUN mv nuflood/build/output/kurganov_petrova /usr/bin/kurganov_petrova
RUN mv nuflood/build/output/flood_fill /usr/bin/flood_fill
RUN rm -r nuflood/
