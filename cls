[33mcommit 66cd5f8e82623a8b667e53b2e39d12b8f130838b[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m
Author: robinmaker123-ai <robinmaker123@gmail.com>
Date:   Fri Oct 9 20:12:37 2026 +0530

    Initial commit: Local Codex AI coding agent

 .gitignore                                 |   17 [32m+[m
 __pycache__/main.cpython-314.pyc           |  Bin [31m0[m -> [32m14260[m bytes
 agent/__pycache__/context.cpython-314.pyc  |  Bin [31m0[m -> [32m505[m bytes
 agent/__pycache__/groq.cpython-314.pyc     |  Bin [31m0[m -> [32m2975[m bytes
 agent/__pycache__/llm.cpython-314.pyc      |  Bin [31m0[m -> [32m2978[m bytes
 agent/__pycache__/loop.cpython-314.pyc     |  Bin [31m0[m -> [32m20723[m bytes
 agent/__pycache__/ollama.cpython-314.pyc   |  Bin [31m0[m -> [32m2351[m bytes
 agent/__pycache__/planner.cpython-314.pyc  |  Bin [31m0[m -> [32m736[m bytes
 agent/context.py                           |   16 [32m+[m
 agent/groq.py                              |   90 [32m+++[m
 agent/llm.py                               |   98 [32m+++[m
 agent/loop.py                              | 1034 [32m++++++++++++++++++++++++++++[m
 agent/ollama.py                            |   60 [32m++[m
 agent/planner.py                           |   25 [32m+[m
 codex.bat                                  |    2 [32m+[m
 config/config.json                         |   20 [32m+[m
 main.py                                    |  353 [32m++++++++++[m
 package-lock.json                          |    6 [32m+[m
 prompts/system.txt                         |    0
 tools/__pycache__/files.cpython-314.pyc    |  Bin [31m0[m -> [32m4039[m bytes
 tools/__pycache__/git.cpython-314.pyc      |  Bin [31m0[m -> [32m113[m bytes
 tools/__pycache__/registry.cpython-314.pyc |  Bin [31m0[m -> [32m3557[m bytes
 tools/__pycache__/search.cpython-314.pyc   |  Bin [31m0[m -> [32m1457[m bytes
 tools/__pycache__/terminal.cpython-314.pyc |  Bin [31m0[m -> [32m1308[m bytes
 tools/files.py                             |  160 [32m+++++[m
 tools/git.py                               |    0
 tools/registry.py                          |  174 [32m+++++[m
 tools/search.py                            |   57 [32m++[m
 tools/terminal.py                          |   62 [32m++[m
 29 files changed, 2174 insertions(+)
