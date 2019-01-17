#!/bin/bash

#for filename in ../Tria_et_al_data/eukaryotes/ingroup/phyml/*.nwk; do
for filename in ../Tria_et_al_data/eukaryotes/ingroup/phyml/*.pruned; do
    ~/workspace/mad/mad.osx $filename
    wait
done

