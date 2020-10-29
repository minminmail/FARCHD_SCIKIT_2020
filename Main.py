# /***********************************************************************
#
# 	This file is part of KEEL-software, the Data Mining tool for regression,
# 	classification, clustering, pattern mining and so on.
#
# 	Copyright (C) 2004-2010
#
# 	F. Herrera (herrera@decsai.ugr.es)
# 	L. S谩nchez (luciano@uniovi.es)
# 	J. Alcal谩-Fdez (jalcala@decsai.ugr.es)
# 	S. Garc铆a (sglopez@ujaen.es)
# 	A. Fernandez (alberto.fernandez@ujaen.es)
# 	J. Luengo (julianlm@decsai.ugr.es)
#
# 	This program is free software: you can redistribute it and/or modify
# 	it under the terms of the GNU General Public License as published by
# 	the Free Software Foundation, either version 3 of the License, or
# 	(at your option) any later version.
#
# 	This program is distributed in the hope that it will be useful,
# 	but WITHOUT ANY WARRANTY; without even the implied warranty of
# 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# 	GNU General Public License for more details.
#
# 	You should have received a copy of the GNU General Public License
# 	along with this program.  If not, see http://www.gnu.org/licenses/
#
# **********************************************************************
from Cython.Compiler import Main

from FarcHD_py.FarcHDClassifier import FarcHDClassifier
from os import listdir
from os.path import isfile, join
import sys
from pathlib import Path

# * <p>It reads the configuration file (data-set files and parameters) and launch the algorithm</p>
# *
# * @author Written by Alberto Fern谩ndez (University of Granada) 14/10/2007
# * @version 1.0
# * @since JDK1.5
from FarcHD_py.LoadFiles import LoadFiles
import numpy as np


class Main:


    # * Main Program
    # * @param args It contains the name of the configuration file
    # * Format:
    # * algorithm = ;algorithm name>
    # * inputData = "training file" "validation file" "test file"
    # * outputData = "training file" "test file"
    # *
    # * seed = value (if used)
    # Parameter1; value1
    # Parameter2&gt; value2

    if __name__ == "__main__":
        # print("Executing Algorithm.")

        # print("sys.argv: " + sys.argv[1])
        # execute(sys.argv[1])
        np.random.seed(0)

        lf = LoadFiles()
        lf.parse_configuration_file("iris", "config0s0.txt")
        X = lf.get_X()
        y = lf.get_y()
        indices = np.random.permutation(len(X))
        iris_X_test = X[indices[-10:]]
        iris_y_test = y[indices[-10:]]
        farchd_classifier = FarcHDClassifier(lf)
        farchd_classifier.fit(X,y)
        test_x = [4.6, 3.1, 1.5, 0.2]
        farchd_classifier.predict(iris_X_test)
        farchd_classifier.score(iris_X_test,iris_y_test)
