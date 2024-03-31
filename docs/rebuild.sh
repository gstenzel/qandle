#!/bin/bash
sphinx-apidoc --separate -a -o ./source ../qandle ../qandle/test
sphinx-build -M html . build