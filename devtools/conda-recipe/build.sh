#!/bin/bash

export ORIG=`pwd`
cd && git clone https://github.com/dynamiq-md/dynamiq_engine
cd dynamiq_engine && python setup.py install
cd $ORIG

$PYTHON setup.py install
