#  Copyright (c) 2019 Humberto Munoz Bauza.
#  Licensed under the MIT License. See LICENSE.txt for license terms.
#  This software is based partially on work supported by IARPA.
#  The U.S. Government is authorized to reproduce and distribute this software for
#  Governmental purposes notwithstanding the conditions of the stated license.
#

from typing import List, Dict

AdjacencyList = List[ Dict[int, float]]

def read_adjacency(filename) -> AdjacencyList:
    #d = {}
    li = []

    with open(filename) as f:
        for l, line in enumerate(f):
            toks = line.split()
            try:
                if len(toks) != 3:
                    raise ValueError()
                i, j, K = int(toks[0]), int(toks[1]), float(toks[2])
                m = max(i, j) + 1 # max num of qubits, zero indexed
                if len(li) < m:
                    for _ in range(m - len(li)):
                        li.append({})
                #d.setdefault(i, {})[j] = K
                li[i][j] = K

            except ValueError:
                raise
    return li



