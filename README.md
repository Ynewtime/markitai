# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| packages/markitai/src/markitai/\_\_init\_\_.py                             |       13 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/\_\_main\_\_.py                             |        2 |        2 |        0 |        0 |      0% |       3-5 |
| packages/markitai/src/markitai/api.py                                      |      239 |       24 |       80 |       15 |     87% |225, 231-232, 283-\>294, 291-\>294, 314, 348, 361-364, 380-381, 382-\>387, 385-\>387, 420-421, 433-434, 439-440, 446-449, 628-630, 632-\>exit, 639-640 |
| packages/markitai/src/markitai/batch.py                                    |      769 |       88 |      274 |       47 |     86% |120-\>119, 122, 125, 128, 273-275, 397, 604, 606, 614-619, 680-681, 697-\>701, 706-\>exit, 730-\>733, 735-\>exit, 823, 854-855, 859-862, 879, 884, 887, 901-902, 942-943, 950, 953-955, 963-\>947, 971-983, 992, 1014, 1024-\>1030, 1032-1033, 1060-\>1066, 1082, 1093-1094, 1108-\>1086, 1116-1117, 1126-\>exit, 1142, 1151-1152, 1184, 1294-\>1303, 1313-1315, 1319-1324, 1333-1337, 1366-1367, 1396, 1441-1442, 1450, 1455-1458, 1469, 1486, 1544, 1550-1552, 1561-1562, 1573, 1577-1580 |
| packages/markitai/src/markitai/cli/\_\_init\_\_.py                         |       19 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/commands/\_\_init\_\_.py                |       14 |        8 |        2 |        0 |     38% |     36-43 |
| packages/markitai/src/markitai/cli/commands/auth.py                        |      225 |       21 |       96 |       20 |     87% |63-65, 68, 80-81, 83, 86, 89, 92, 161, 171, 183, 219, 224, 227, 286-\>288, 338, 385, 388-\>403, 391-\>399, 461, 474, 527 |
| packages/markitai/src/markitai/cli/commands/cache.py                       |      158 |        4 |       64 |        7 |     95% |44-\>56, 56-\>exit, 68-\>73, 70-\>73, 177, 237-238, 253-\>exit, 293 |
| packages/markitai/src/markitai/cli/commands/config.py                      |      269 |       43 |      100 |       15 |     83% |42, 65, 67, 72-73, 75, 78, 85, 159-\>161, 176, 200-202, 206-207, 209-212, 216, 289-291, 292-\>exit, 307, 434, 436, 487-499, 508-512, 525-527 |
| packages/markitai/src/markitai/cli/commands/doctor.py                      |      377 |       36 |      150 |       19 |     90% |99, 119-122, 129, 130-\>135, 156-158, 195-199, 245-246, 258, 298-\>308, 349, 425-426, 436-437, 475-476, 592-\>588, 695-703, 731-747, 759, 788, 816, 860, 921-\>925, 925-\>929, 958-\>945, 1141-1142 |
| packages/markitai/src/markitai/cli/commands/init.py                        |      231 |       63 |       72 |       11 |     71% |75-77, 204-224, 228-230, 239-247, 256-267, 276-283, 309-312, 320-\>319, 326-340, 344-351, 354-\>365, 414-\>442, 419-\>416, 426, 428, 483 |
| packages/markitai/src/markitai/cli/commands/mcp.py                         |       13 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/commands/serve.py                       |      134 |       10 |       42 |        6 |     91% |91-92, 104-\>106, 157, 162-168, 185-\>exit, 187-\>192, 190-191, 293-300, 311 |
| packages/markitai/src/markitai/cli/config\_editor.py                       |      308 |      154 |      110 |       11 |     50% |66-\>72, 85-87, 94, 134-\>51, 183-361, 373, 419-\>423, 451, 456-462, 474, 491-492, 497-523, 548, 554, 560, 581-582 |
| packages/markitai/src/markitai/cli/console.py                              |        3 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/framework.py                            |       85 |        3 |       34 |        4 |     94% |124, 158, 206-\>208, 226-\>228, 263 |
| packages/markitai/src/markitai/cli/hints.py                                |        9 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/i18n.py                                 |       29 |        2 |       10 |        0 |     95% |   191-193 |
| packages/markitai/src/markitai/cli/interactive.py                          |      243 |       82 |       84 |       23 |     63% |36-43, 52-54, 78, 109, 123, 136-145, 164-167, 170-\>179, 174-177, 187-\>196, 189-190, 202, 230-245, 251, 270-271, 274, 277, 300-352, 364-\>371, 379-406, 420, 429-430, 433, 466-467, 471, 492-499, 501, 503, 514-\>517 |
| packages/markitai/src/markitai/cli/logging\_config.py                      |      171 |       12 |       52 |        8 |     91% |39-\>exit, 152, 164-165, 189-190, 327, 331, 430, 433, 491, 542, 557 |
| packages/markitai/src/markitai/cli/main.py                                 |      482 |       89 |      196 |       12 |     81% |17-20, 169, 183-193, 210, 221-222, 230-231, 636-655, 665-670, 691-694, 830, 930-932, 1065-1097, 1100, 1164-1224, 1277-1278, 1325-1331, 1339-1340, 1349 |
| packages/markitai/src/markitai/cli/processors/\_\_init\_\_.py              |       35 |        8 |       14 |        4 |     76% |74-76, 83-85, 87-89, 95-97 |
| packages/markitai/src/markitai/cli/processors/batch.py                     |      485 |       42 |      170 |       23 |     90% |66-67, 391-392, 405-406, 442-\>446, 471-472, 544-556, 680, 693-696, 820-\>815, 822-823, 835-\>837, 841, 888, 920, 929-\>931, 953, 983-\>1011, 1038, 1064, 1070-1075, 1098, 1110-\>1117, 1115-1116, 1134, 1140-1145, 1174-\>1210, 1198-1199, 1212-\>1251, 1232-1234, 1254-\>1252, 1258-\>1256 |
| packages/markitai/src/markitai/cli/processors/batch\_llm.py                |      313 |       95 |      112 |       16 |     66% |120-122, 152, 169, 339, 341-343, 357-359, 397-406, 409-\>411, 413, 418-513, 595-610, 668, 688-693, 701, 708, 719, 732, 752, 779-812 |
| packages/markitai/src/markitai/cli/processors/file.py                      |      164 |       14 |       62 |       14 |     88% |56, 58, 109-111, 197, 206-\>217, 249-251, 302-\>320, 310, 345-\>428, 365-366, 367-\>373, 385, 393, 398, 405-\>428, 413-\>423 |
| packages/markitai/src/markitai/cli/processors/llm.py                       |      155 |        2 |       52 |        6 |     96% |148-\>145, 157-\>164, 312-\>363, 350, 358, 451-\>454 |
| packages/markitai/src/markitai/cli/processors/url.py                       |      569 |       97 |      172 |       33 |     79% |86, 189, 329, 339-\>341, 361-365, 389-390, 397, 403-422, 469-479, 492, 536-554, 566-602, 622-\>637, 664-681, 686, 690-698, 702-704, 716-718, 730-741, 752-755, 769, 862, 866, 1099-1101, 1106, 1122-1131, 1159-1161, 1195, 1274, 1338-1343, 1392, 1409-\>1401, 1571-1572, 1614-1615, 1720-1734 |
| packages/markitai/src/markitai/cli/processors/validators.py                |      108 |        8 |       48 |        1 |     93% |103-111, 188-189 |
| packages/markitai/src/markitai/cli/providers\_detect.py                    |       71 |       11 |       20 |        1 |     87% |38-45, 50, 55, 65-66, 150-\>161 |
| packages/markitai/src/markitai/cli/ui.py                                   |      264 |        5 |       84 |        4 |     97% |436-\>exit, 464, 467, 527-528, 614 |
| packages/markitai/src/markitai/config.py                                   |      488 |       37 |      138 |       26 |     89% |650, 652, 735, 781, 790, 916, 919, 921, 924, 976, 1058-\>1070, 1077, 1096, 1112, 1143, 1147, 1150, 1156, 1159, 1199-1202, 1205, 1212-1214, 1225, 1228, 1250, 1269-1278, 1281-\>exit, 1286-1290 |
| packages/markitai/src/markitai/constants.py                                |      101 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/\_\_init\_\_.py                   |       12 |        6 |        2 |        0 |     43% |   131-137 |
| packages/markitai/src/markitai/converter/\_patches.py                      |       47 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/base.py                           |      111 |        1 |        6 |        0 |     99% |       242 |
| packages/markitai/src/markitai/converter/cloudflare.py                     |       55 |        4 |       12 |        4 |     88% |121-\>123, 124-128, 136, 169 |
| packages/markitai/src/markitai/converter/delimited.py                      |       44 |        5 |        8 |        0 |     90% |53-54, 86-88 |
| packages/markitai/src/markitai/converter/eml.py                            |      133 |       19 |       46 |       10 |     83% |65, 68, 77-79, 85, 103-119, 129-\>124, 139, 146-148, 187-188, 219-\>222, 265-\>270, 267-\>270 |
| packages/markitai/src/markitai/converter/heif.py                           |       29 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/image.py                          |       93 |        4 |       30 |        2 |     95% |   227-233 |
| packages/markitai/src/markitai/converter/latex.py                          |      142 |       10 |       56 |        9 |     89% |79-\>74, 109, 122-123, 128, 130-133, 140, 151, 174-\>176, 179-\>181 |
| packages/markitai/src/markitai/converter/legacy.py                         |       31 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/markitdown\_ext.py                |       72 |        3 |        8 |        0 |     96% |74-75, 199 |
| packages/markitai/src/markitai/converter/markup.py                         |      131 |        2 |       36 |        2 |     98% |64-\>78, 73-74 |
| packages/markitai/src/markitai/converter/office.py                         |      223 |       80 |       52 |        9 |     63% |73, 140-157, 188-\>192, 198-\>205, 233-252, 258-354, 377-383, 401-406, 422-423, 450-457, 461-462, 550-\>554 |
| packages/markitai/src/markitai/converter/opendocument.py                   |      168 |       14 |       68 |       11 |     89% |54-55, 66-67, 87, 95, 104-105, 122, 142, 149-\>143, 157-\>exit, 160-161, 165-\>exit, 170-\>exit, 174-\>exit, 178-179, 206-\>211, 252-\>255 |
| packages/markitai/src/markitai/converter/pdf.py                            |      588 |       47 |      206 |       27 |     90% |128-\>124, 147, 157, 178, 183-184, 281, 319-321, 395-398, 407, 444, 448, 558, 589-\>677, 593-\>591, 644-645, 701-\>704, 778, 798, 885, 950-952, 983, 986-989, 990-\>993, 1035-\>1033, 1078-1079, 1098-\>1102, 1120-1122, 1180-1182, 1219-1221, 1231-\>1233, 1233-\>1235, 1268-1270, 1346-1349, 1353-\>1361, 1396-\>1408, 1410, 1413 |
| packages/markitai/src/markitai/converter/rtf.py                            |      640 |       41 |      290 |       31 |     92% |384, 418-\>420, 427-428, 434, 436, 437-\>441, 450, 452-458, 460-461, 530-531, 537-539, 610, 618, 624, 634, 635-\>exit, 648-\>exit, 666, 678-\>exit, 683-\>exit, 692, 702-\>exit, 709, 719, 733, 756-\>exit, 764-766, 831-832, 833-\>836, 876, 944-\>946, 950, 1028-1030, 1037-1039 |
| packages/markitai/src/markitai/converter/text.py                           |       15 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/xml\_doc.py                       |       50 |        1 |       14 |        1 |     97% |        60 |
| packages/markitai/src/markitai/domain\_profiles.py                         |        4 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/fetch.py                                    |      380 |       41 |      168 |       20 |     87% |342-346, 371-372, 412, 440, 443-\>445, 487-\>495, 507, 522, 585-587, 594, 599, 631-\>633, 666-\>668, 701-720, 726, 730-\>733, 867-868, 1021, 1057, 1061-\>1063, 1068-1071, 1080-1082, 1100-1102, 1115-1117 |
| packages/markitai/src/markitai/fetch\_cache.py                             |      251 |        6 |       50 |        2 |     97% |40, 45, 492-493, 551-552 |
| packages/markitai/src/markitai/fetch\_consent.py                           |      107 |        4 |       38 |        4 |     94% |72, 140, 158, 190 |
| packages/markitai/src/markitai/fetch\_http.py                              |      225 |       55 |       66 |       11 |     73% |20-22, 70-\>87, 85, 103-105, 107, 204-205, 211-216, 233, 237, 267-268, 280-284, 292, 348-349, 353, 381-410, 427, 446-458, 479-\>481, 483 |
| packages/markitai/src/markitai/fetch\_playwright.py                        |      454 |       72 |      170 |       31 |     80% |110, 115, 128-\>123, 129-\>123, 139-174, 285-286, 457, 502-506, 511-518, 622, 625, 642, 654, 675-676, 681, 686-\>695, 691-692, 698-699, 705-706, 718-\>760, 746, 749-\>752, 767-773, 774-\>812, 776-\>812, 803-810, 853-856, 924-926, 968, 970, 972, 974, 976, 978, 980, 982, 984, 1142-1143, 1212 |
| packages/markitai/src/markitai/fetch\_policy.py                            |      247 |       28 |      124 |       16 |     88% |56, 65, 79, 99, 185, 196, 219, 223, 225, 253-254, 257, 272-273, 298-303, 311-312, 325-326, 329, 341, 346-347, 350, 380, 385 |
| packages/markitai/src/markitai/fetch\_screenshot.py                        |       66 |        8 |       16 |        1 |     89% |70-71, 74-77, 159-161 |
| packages/markitai/src/markitai/fetch\_session.py                           |      362 |       84 |      156 |        9 |     72% |54-\>41, 161-\>159, 195-\>180, 223-264, 267-329, 478-479, 490-496, 504, 586-591, 599, 712-715, 721-722, 769 |
| packages/markitai/src/markitai/fetch\_strategies/\_\_init\_\_.py           |       34 |        0 |        6 |        2 |     95% |70-\>exit, 72-\>exit |
| packages/markitai/src/markitai/fetch\_strategies/\_shared.py               |       45 |        2 |        8 |        3 |     91% |57, 84, 94-\>97 |
| packages/markitai/src/markitai/fetch\_strategies/cloudflare.py             |      106 |        9 |       42 |       10 |     87% |95-\>97, 97-\>99, 99-\>105, 154-\>198, 199, 220-221, 226, 248, 251, 265, 275-280 |
| packages/markitai/src/markitai/fetch\_strategies/defuddle.py               |       64 |       18 |       20 |        4 |     64% |96, 109-119, 123-126, 129, 141-144 |
| packages/markitai/src/markitai/fetch\_strategies/jina.py                   |       84 |        7 |       30 |        8 |     87% |40-\>38, 42, 112, 114, 116, 141, 162-\>167, 185-186 |
| packages/markitai/src/markitai/fetch\_strategies/playwright.py             |       34 |        2 |       14 |        2 |     92% |    31, 44 |
| packages/markitai/src/markitai/fetch\_strategies/static.py                 |      134 |       19 |       62 |        7 |     83% |47-\>53, 53-\>60, 74-\>81, 85-86, 94-103, 121, 278, 337-347 |
| packages/markitai/src/markitai/fetch\_support.py                           |       42 |        2 |       16 |        1 |     95% |   121-122 |
| packages/markitai/src/markitai/fetch\_types.py                             |       40 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/image.py                                    |      624 |       75 |      184 |       23 |     86% |108, 111, 182-199, 214-249, 274, 333, 349-351, 524, 581, 588-595, 604-\>607, 613, 624-631, 722, 822-824, 942-944, 985-986, 1003, 1062, 1071, 1145-1147, 1176-1177, 1194, 1431-1433, 1450, 1452, 1487-\>1519, 1515-1516 |
| packages/markitai/src/markitai/json\_order.py                              |      175 |       24 |      104 |       11 |     82% |226, 231-\>239, 317-\>321, 384-\>388, 398-\>404, 411-423, 448-\>467, 456-463, 467-\>483, 473-480, 502-\>519 |
| packages/markitai/src/markitai/llm/\_\_init\_\_.py                         |        7 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/batch\_api.py                           |      170 |       44 |       40 |        5 |     71% |114-115, 244, 361, 389, 392, 396, 415-420, 432-435, 457-475, 487-508 |
| packages/markitai/src/markitai/llm/cache.py                                |      255 |        7 |       76 |        3 |     96% |409-410, 414-415, 504-\>511, 535-\>exit, 538-539, 549 |
| packages/markitai/src/markitai/llm/content.py                              |      283 |       24 |      144 |       13 |     89% |71, 86-\>82, 288, 297-\>300, 306-335, 346-\>344, 348-\>355, 361-\>360, 364-\>360, 368-\>360, 376-380, 411-\>447, 632 |
| packages/markitai/src/markitai/llm/degeneration.py                         |       68 |        0 |       32 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/document.py                             |      586 |       15 |      126 |       11 |     96% |219-220, 471, 561, 593-594, 615, 618, 631, 637, 690, 1268-1272, 1330-\>1361, 1901, 1962-\>1964 |
| packages/markitai/src/markitai/llm/engine.py                               |      354 |       22 |      124 |       13 |     92% |125-\>127, 127-\>exit, 241-243, 285-\>283, 401, 504, 598, 701-\>714, 722-\>729, 928-929, 1056-1064, 1084-1105, 1137 |
| packages/markitai/src/markitai/llm/models.py                               |       73 |        0 |       22 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/processor.py                            |      463 |       73 |      154 |       17 |     82% |33-34, 221, 326-\>335, 525, 609-610, 640-646, 673, 696, 726-730, 856-859, 877-\>875, 879-\>881, 881-\>875, 887-\>893, 954-956, 1087, 1108-\>1114, 1111-\>1114, 1115, 1152-1170, 1187-1188, 1197-1254, 1273-\>1277 |
| packages/markitai/src/markitai/llm/router.py                               |      175 |        1 |       64 |        3 |     98% |205-\>212, 248, 502-\>514 |
| packages/markitai/src/markitai/llm/structured.py                           |       64 |        2 |       24 |        3 |     94% |118, 140, 145-\>138 |
| packages/markitai/src/markitai/llm/types.py                                |       81 |        5 |       16 |        5 |     90% |156, 159, 169, 171, 176 |
| packages/markitai/src/markitai/llm/vision.py                               |      381 |       19 |      114 |       16 |     93% |159, 167, 222, 225, 235, 376, 681-683, 853-\>870, 870-\>878, 961, 963, 981, 983, 990-994, 1084-\>1089, 1214-\>1248, 1228-\>1230, 1241-1242, 1248-\>1282, 1262-\>1264, 1275-1276 |
| packages/markitai/src/markitai/mcp/\_\_init\_\_.py                         |        0 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/mcp/server.py                               |      136 |        1 |       20 |        0 |     99% |       571 |
| packages/markitai/src/markitai/ocr.py                                      |      273 |       55 |       80 |       16 |     77% |60-61, 139-\>141, 142-\>144, 149-\>151, 151-\>exit, 211-\>221, 271-272, 285-\>304, 370-372, 385-392, 406, 408, 419, 448, 478-479, 485, 511, 516, 529-544, 563-587, 611-622, 639-640, 693-695 |
| packages/markitai/src/markitai/output\_profiles.py                         |      178 |        7 |       76 |        9 |     94% |108, 156-\>158, 158-\>160, 166-\>168, 168-\>170, 252, 268-269, 322-\>320, 395-\>413, 402-409, 410-\>413 |
| packages/markitai/src/markitai/ports.py                                    |       27 |        3 |        2 |        0 |     90% |22, 26, 32 |
| packages/markitai/src/markitai/prompts/\_\_init\_\_.py                     |       86 |       12 |       40 |        4 |     83% |151-\>155, 167, 182-\>185, 209-221 |
| packages/markitai/src/markitai/providers/\_\_init\_\_.py                   |      310 |       46 |      118 |       14 |     84% |138, 150-151, 192, 227-229, 271, 312-\>321, 360-361, 421-427, 436-442, 480-486, 501-507, 519-522, 524-\>530, 575, 578, 614-616, 644-647, 691, 702-704, 738 |
| packages/markitai/src/markitai/providers/auth.py                           |      343 |       88 |      110 |        6 |     74% |52-53, 267-275, 285-302, 311-315, 336, 357-388, 393-428, 433-440, 516-\>520, 559-560, 635-639, 647-648, 847, 864-872, 913-914, 925 |
| packages/markitai/src/markitai/providers/chatgpt.py                        |      175 |       40 |       64 |       12 |     72% |55-57, 69-71, 94, 123-124, 133, 181-182, 201, 204, 207-208, 215-\>199, 225-229, 242-243, 275-278, 287, 293-\>285, 301, 342-\>351, 357, 369-370, 375-376, 416, 487-493 |
| packages/markitai/src/markitai/providers/claude\_agent.py                  |      179 |       22 |       76 |       16 |     85% |74-78, 107, 169, 174-\>167, 193-197, 229-234, 238-243, 269-\>exit, 312-\>315, 325, 358, 361, 387, 394-\>382, 464-\>463, 466-\>461, 468-\>472, 493-496, 580 |
| packages/markitai/src/markitai/providers/common.py                         |       37 |        8 |       22 |        1 |     81% |27-\>24, 100-115 |
| packages/markitai/src/markitai/providers/copilot.py                        |      317 |      107 |      100 |       26 |     62% |81-82, 94-98, 114-174, 217, 236-237, 241-278, 309, 311, 316-\>315, 320-\>315, 357-366, 370, 372, 380-\>378, 382-383, 406-410, 422, 465, 469-470, 495, 497-\>537, 506, 571, 574, 583, 608-609, 625-628, 654, 666, 670-674, 677-678, 684, 706-709, 715-716, 731-\>736, 771-\>781, 805, 810-\>817, 813-814 |
| packages/markitai/src/markitai/providers/discovery.py                      |      320 |       48 |      136 |       25 |     82% |51-55, 92, 171-172, 254, 313, 329-373, 384, 402, 406-\>408, 409-\>411, 426, 445-449, 454, 484-\>560, 491-\>497, 502-\>504, 504-\>560, 511-\>513, 531, 535-\>560, 548-\>560, 572, 577-\>570, 585-\>583, 593-\>589, 606, 647, 657-659, 692 |
| packages/markitai/src/markitai/providers/errors.py                         |       43 |        1 |        8 |        2 |     94% |284, 305-\>exit |
| packages/markitai/src/markitai/providers/oauth\_display.py                 |       53 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/providers/timeout.py                        |       48 |        2 |       26 |        5 |     91% |138, 143-\>135, 147, 152-\>145, 154-\>145 |
| packages/markitai/src/markitai/runs/\_\_init\_\_.py                        |        5 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/runs/history.py                             |      105 |        9 |       36 |        4 |     91% |59-60, 67, 78, 102, 207, 262-264 |
| packages/markitai/src/markitai/runs/json\_output.py                        |       16 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/runs/output.py                              |       82 |        6 |       22 |        2 |     92% |139-140, 226-\>234, 231-232, 244-245 |
| packages/markitai/src/markitai/runs/report.py                              |       32 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/runs/types.py                               |       22 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/security.py                                 |      158 |       33 |       50 |        8 |     75% |39-51, 69-81, 126-\>132, 129-130, 138, 197, 244, 263-264, 354, 369, 372-373 |
| packages/markitai/src/markitai/serve/\_\_init\_\_.py                       |       15 |        2 |        2 |        0 |     88% |     68-69 |
| packages/markitai/src/markitai/serve/app.py                                |     1307 |      123 |      480 |       82 |     88% |166-172, 191-196, 214-\>229, 240-241, 251-252, 259, 271, 321-\>369, 387-388, 523, 525, 531, 605, 635-636, 667-\>673, 692-\>694, 695, 699, 750-759, 836, 861, 864, 867-868, 904, 912, 967, 1024, 1049-1050, 1091, 1094, 1114, 1117, 1155, 1252-1258, 1265, 1305, 1340, 1361-1364, 1371, 1381, 1398, 1401, 1418-1419, 1435, 1437, 1481-1482, 1531, 1569-1570, 1594-1597, 1796, 1862-1880, 1911, 1926, 1978-1983, 1988-1993, 2026, 2034, 2082, 2096, 2098, 2110, 2217, 2224, 2253, 2279, 2299, 2312-\>2314, 2333-2339, 2341-\>2346, 2343, 2346-\>2352, 2348, 2355, 2358, 2370, 2371-\>2376, 2373, 2376-\>2381, 2378, 2404, 2416-\>2418, 2437-\>2442, 2445-2446, 2453, 2460, 2521-2522, 2773, 2786-\>2795, 2788-\>2795, 2791-2794, 2828-2830, 2850, 2857, 2869-2872, 2878-\>exit |
| packages/markitai/src/markitai/serve/jobs.py                               |      484 |       95 |      118 |       19 |     79% |49-51, 264-\>exit, 289, 291-294, 304, 311-315, 322-323, 409, 427, 477-480, 491-494, 505-506, 508, 512-513, 562, 565-566, 575-578, 593-594, 613-614, 669, 713-714, 761-766, 844-858, 889-901, 911-912, 955, 957-\>exit, 970-\>988, 982-\>988, 984-986, 992-1004 |
| packages/markitai/src/markitai/serve/openapi.py                            |       18 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/serve/schemas.py                            |      314 |        4 |       20 |        4 |     98% |60, 99, 164, 236 |
| packages/markitai/src/markitai/types.py                                    |       10 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/urls.py                                     |       75 |       12 |       36 |        7 |     83% |84-85, 88, 98, 100-101, 114-115, 117-118, 127, 160 |
| packages/markitai/src/markitai/utils/\_\_init\_\_.py                       |        9 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/asset\_store.py                       |       47 |        1 |       14 |        3 |     93% |29, 84-\>90, 87-\>84 |
| packages/markitai/src/markitai/utils/cli\_helpers.py                       |       64 |        3 |       18 |        3 |     93% |80-81, 89-\>100, 103, 155-\>169 |
| packages/markitai/src/markitai/utils/clock.py                              |        4 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/errors.py                             |       16 |        1 |        0 |        0 |     94% |       100 |
| packages/markitai/src/markitai/utils/executor.py                           |       82 |       25 |       28 |        8 |     66% |49-\>54, 75-97, 103-\>102, 107-119, 130, 134, 136, 140 |
| packages/markitai/src/markitai/utils/frontmatter.py                        |      177 |       10 |       88 |       11 |     92% |32, 93, 106, 119-122, 168-\>170, 252, 256, 264, 269, 347-\>354, 373-\>365 |
| packages/markitai/src/markitai/utils/guidance.py                           |       29 |        0 |        8 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/markdown\_quality.py                  |        9 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/mime.py                               |       42 |        3 |       18 |        3 |     90% |132, 157, 162 |
| packages/markitai/src/markitai/utils/office.py                             |       60 |       33 |       22 |        3 |     44% |21, 44-77, 100-113, 117, 130-131 |
| packages/markitai/src/markitai/utils/office\_mac.py                        |      182 |       21 |       48 |        9 |     87% |203-204, 213-214, 230, 256-257, 263, 273-274, 366, 369, 391-\>393, 406-407, 415-417, 426-427, 435-436 |
| packages/markitai/src/markitai/utils/output.py                             |       31 |        0 |       12 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/paths.py                              |       31 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/shutdown.py                           |       16 |       16 |        2 |        0 |      0% |     25-58 |
| packages/markitai/src/markitai/utils/sqlite\_cache.py                      |       23 |        1 |        6 |        1 |     93% |        49 |
| packages/markitai/src/markitai/utils/suppress.py                           |       19 |        2 |        4 |        1 |     87% |45-\>exit, 47-48 |
| packages/markitai/src/markitai/utils/term.py                               |       48 |        1 |       18 |        2 |     95% |52, 84-\>86 |
| packages/markitai/src/markitai/utils/terminal\_image.py                    |       66 |        1 |       22 |        1 |     98% |        83 |
| packages/markitai/src/markitai/utils/text.py                               |      283 |       18 |      122 |       17 |     90% |133-\>142, 156-\>158, 158-\>160, 166, 204, 276, 291, 366-\>373, 389, 571, 590-597, 631, 683-684, 707-709, 711-\>673, 716 |
| packages/markitai/src/markitai/utils/url\_redaction.py                     |       27 |        5 |        4 |        1 |     81% |20, 25-26, 33-34 |
| packages/markitai/src/markitai/vision\_consent.py                          |       50 |        2 |       16 |        2 |     94% |    80, 92 |
| packages/markitai/src/markitai/webextract/\_\_init\_\_.py                  |       31 |        3 |       10 |        2 |     88% |32, 59, 89 |
| packages/markitai/src/markitai/webextract/constants.py                     |       26 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/content\_boundary.py             |      132 |       12 |       92 |       10 |     89% |54, 69, 88, 144-147, 149, 170, 173-\>172, 181, 192, 194 |
| packages/markitai/src/markitai/webextract/dom.py                           |       84 |       25 |       46 |       10 |     65% |27, 49, 53, 94-98, 100, 104, 109, 118, 139-\>90, 151-165 |
| packages/markitai/src/markitai/webextract/elements/\_\_init\_\_.py         |        5 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/callouts.py             |       71 |        2 |       32 |        7 |     91% |35-\>42, 43-\>47, 55-\>62, 75, 78-\>84, 112, 121-\>124 |
| packages/markitai/src/markitai/webextract/elements/code.py                 |      129 |        7 |       76 |        9 |     92% |133, 136, 150, 181, 186-\>194, 206-\>209, 238, 270, 274 |
| packages/markitai/src/markitai/webextract/elements/footnotes.py            |     1013 |      104 |      562 |      103 |     86% |64-65, 71-72, 117, 127, 129, 136, 143, 163, 167-173, 182-\>178, 190, 194, 258, 272-\>271, 289, 295-\>297, 309-312, 327, 334, 368-\>367, 395, 398, 402-\>404, 420-\>422, 433, 436-\>442, 449, 456-\>451, 473, 480-\>482, 491-\>497, 518, 523-\>530, 554, 579-\>575, 583, 587, 596-\>600, 608-\>613, 639-\>637, 646, 649-\>653, 654, 658-\>665, 663, 676-\>674, 686, 690-\>693, 701-\>683, 720-\>exit, 761-\>759, 787, 821-\>exit, 840-\>849, 843-\>845, 863, 880, 882-\>876, 884-\>876, 939-\>946, 983, 990, 1018, 1029-1030, 1063-\>1060, 1068, 1077-\>1099, 1092-1093, 1129-\>1135, 1140, 1145-\>1150, 1161-\>1167, 1174-\>1173, 1190, 1214-\>1221, 1224-1257, 1271, 1274, 1278, 1284-1287, 1294-\>1267, 1319, 1323, 1362, 1364, 1373, 1376, 1379, 1384-\>1392, 1402-\>1405, 1406, 1453, 1455, 1476, 1479, 1482, 1490, 1492, 1495, 1539-\>1544, 1545, 1552-\>1557, 1565, 1572-1575, 1610 |
| packages/markitai/src/markitai/webextract/elements/headings.py             |       19 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/images.py               |       80 |        2 |       42 |        4 |     95% |29-\>32, 69, 74-\>71, 126 |
| packages/markitai/src/markitai/webextract/elements/math.py                 |      108 |        1 |       60 |        6 |     96% |74-\>63, 97-\>99, 103-\>93, 105-\>93, 182-\>179, 200 |
| packages/markitai/src/markitai/webextract/enrichers/\_\_init\_\_.py        |        3 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/enrichers/base.py                |       15 |        2 |        0 |        0 |     87% |    70, 83 |
| packages/markitai/src/markitai/webextract/enrichers/x\_oembed.py           |      292 |        9 |      122 |       17 |     93% |82-\>87, 103-\>134, 197-\>200, 405, 412, 419-\>418, 426-\>424, 428-\>424, 465, 469, 498-\>479, 516-517, 527, 546-548, 565-\>567, 569-\>572 |
| packages/markitai/src/markitai/webextract/extractors/\_\_init\_\_.py       |        3 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/extractors/base.py               |        8 |        1 |        0 |        0 |     88% |        15 |
| packages/markitai/src/markitai/webextract/extractors/bilibili\_opus.py     |       74 |       15 |       32 |       13 |     68% |51-52, 62, 74-\>76, 76-\>78, 78-\>81, 110-119, 129, 141, 145, 164-\>166, 166-\>168, 168-\>171 |
| packages/markitai/src/markitai/webextract/extractors/github\_repo.py       |       23 |        0 |        8 |        1 |     97% |   90-\>92 |
| packages/markitai/src/markitai/webextract/extractors/github\_thread.py     |       81 |        7 |       32 |        6 |     85% |147-150, 156-\>166, 163-\>166, 187, 233-\>240, 237-238 |
| packages/markitai/src/markitai/webextract/extractors/hackernews\_thread.py |      103 |        8 |       50 |       13 |     85% |59-60, 136-\>142, 138-\>142, 146-\>148, 151, 168, 173-\>178, 199-\>201, 224, 231-\>241, 236-238, 243-\>247, 250-\>255 |
| packages/markitai/src/markitai/webextract/extractors/reddit\_post.py       |      101 |       11 |       48 |       13 |     81% |64-65, 83-\>82, 151-157, 171-\>175, 191-\>194, 211, 240, 247-\>231, 254, 260-\>272, 262-\>265, 266-\>272, 268-\>272 |
| packages/markitai/src/markitai/webextract/extractors/registry.py           |       35 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/extractors/steam\_news.py        |       83 |       10 |       20 |        7 |     83% |40-41, 71-\>73, 73-\>75, 75-\>80, 100, 105-107, 114, 119-121 |
| packages/markitai/src/markitai/webextract/extractors/substack\_note.py     |      147 |       33 |       60 |        5 |     72% |60, 65-66, 67-\>61, 114-\>120, 117-\>120, 226-231, 236-252, 257-263 |
| packages/markitai/src/markitai/webextract/extractors/x\_article.py         |       13 |        1 |        2 |        1 |     87% |        44 |
| packages/markitai/src/markitai/webextract/extractors/x\_common.py          |      356 |       60 |      220 |       54 |     77% |87, 93, 146, 149-\>143, 155-\>159, 157-\>159, 159-\>164, 185, 188, 201, 221, 225-\>219, 244-\>246, 279, 283-285, 290-\>292, 294, 306, 312, 317, 342, 345, 347, 351, 358, 395, 420, 448, 501, 507, 509-\>504, 511-\>504, 513-\>499, 517-535, 565, 569, 578-579, 584-\>581, 589, 623-\>611, 628, 645, 646-\>643, 648, 666, 671, 672-\>675, 700, 703, 706, 710, 726-\>730, 729, 773-775 |
| packages/markitai/src/markitai/webextract/extractors/x\_tweet.py           |      132 |       11 |       72 |       14 |     87% |71, 116, 129-\>152, 138, 147-\>136, 150, 171-\>174, 174-\>182, 180-181, 185-\>188, 204, 247, 261-262, 267-\>265, 276 |
| packages/markitai/src/markitai/webextract/extractors/youtube\_page.py      |       97 |       29 |       56 |       17 |     58% |57-58, 83-\>85, 85-\>87, 87-\>90, 122-\>126, 135-151, 172-179, 199-210, 232-\>235, 235-\>238, 240-\>251, 243-\>251 |
| packages/markitai/src/markitai/webextract/frontmatter.py                   |       14 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/html\_to\_markdown.py            |      281 |       25 |      158 |       29 |     88% |57, 62, 174, 177, 187, 209, 220, 228, 237, 245, 257, 278, 286, 295, 327-\>324, 330, 347, 372, 374, 381, 384, 392, 409, 427-\>429, 432-\>440, 438-\>440, 446, 506, 512 |
| packages/markitai/src/markitai/webextract/markdown.py                      |      117 |        9 |       48 |       10 |     88% |90, 95-\>97, 144, 147, 189, 197, 205, 237, 243, 259 |
| packages/markitai/src/markitai/webextract/metadata.py                      |      163 |       10 |      102 |       13 |     91% |60, 115, 172, 176-\>179, 181, 205, 238, 241-242, 254, 257-\>256, 261-\>exit, 263-\>262, 279, 282-\>276 |
| packages/markitai/src/markitai/webextract/mobile\_styles.py                |       58 |        5 |       34 |        5 |     89% |28, 38-\>37, 41-42, 68, 72, 81-\>74 |
| packages/markitai/src/markitai/webextract/pipeline.py                      |      217 |        8 |       72 |       10 |     93% |107-\>106, 116, 139-140, 299-\>308, 309-\>320, 320-\>322, 416-\>420, 418-\>420, 511, 544-546, 548 |
| packages/markitai/src/markitai/webextract/preprocess.py                    |       54 |        1 |       14 |        1 |     97% |       133 |
| packages/markitai/src/markitai/webextract/quality.py                       |      130 |        3 |       48 |        2 |     97% |88, 136-137 |
| packages/markitai/src/markitai/webextract/removals/\_\_init\_\_.py         |       19 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/removals/content\_patterns.py    |      895 |      119 |      502 |       60 |     85% |252, 263-274, 298, 302, 329-\>328, 336, 366, 370, 441, 447, 450, 452, 463, 491-492, 506, 513, 525-526, 530, 551-559, 570, 577-578, 585-586, 588-589, 635-\>637, 675-\>682, 691-695, 722-\>729, 745-749, 775, 784-785, 789-791, 795, 811, 849, 861-862, 898-899, 920-921, 929, 935, 946, 955-\>978, 960-977, 981-983, 992, 1003-\>998, 1014-1026, 1042-\>1058, 1094, 1103-1104, 1142, 1153, 1158, 1160, 1171, 1194, 1204, 1208, 1233, 1242-\>1247, 1246, 1256, 1267, 1317, 1361 |
| packages/markitai/src/markitai/webextract/removals/hidden.py               |       64 |        9 |       42 |        9 |     83% |51, 83, 103, 111, 115, 117, 122, 124, 128 |
| packages/markitai/src/markitai/webextract/removals/scoring.py              |      108 |       10 |       66 |        5 |     90% |78, 80-81, 89, 109-110, 130, 168-\>174, 193-195 |
| packages/markitai/src/markitai/webextract/removals/selectors.py            |      147 |       28 |       94 |        8 |     81% |55, 62-77, 97, 100, 120-121, 143-144, 153-\>157, 168, 182-184, 211, 216 |
| packages/markitai/src/markitai/webextract/removals/small\_images.py        |       66 |        1 |       38 |        4 |     95% |78-\>86, 80-\>82, 82-\>86, 102 |
| packages/markitai/src/markitai/webextract/render.py                        |      128 |        3 |       64 |        6 |     95% |106, 159, 162, 188-\>194, 213-\>215, 215-\>217 |
| packages/markitai/src/markitai/webextract/resolver.py                      |       36 |        0 |       12 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/sanitize.py                      |       26 |        0 |       18 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/schema.py                        |       38 |        4 |       20 |        2 |     90% |23, 26-27, 43 |
| packages/markitai/src/markitai/webextract/scoring.py                       |      133 |       14 |       66 |        4 |     89% |72, 86, 157-158, 173-180, 203, 217 |
| packages/markitai/src/markitai/webextract/semantics.py                     |       38 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/standardize.py                   |      311 |       30 |      190 |       28 |     88% |129, 136-140, 158, 161-\>186, 164, 172-\>174, 175, 177, 183, 185, 202-\>213, 204-208, 215-216, 233, 245, 272-274, 282-\>261, 285-\>261, 287-\>261, 306, 371, 394, 424, 461, 463, 518, 541, 569-\>567 |
| packages/markitai/src/markitai/webextract/thread\_policy.py                |       16 |        1 |        4 |        1 |     90% |        62 |
| packages/markitai/src/markitai/webextract/types.py                         |       44 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/utils.py                         |       22 |        0 |        2 |        0 |    100% |           |
| packages/markitai/src/markitai/workflow/\_\_init\_\_.py                    |       15 |        4 |        6 |        2 |     71% |30-32, 34-36 |
| packages/markitai/src/markitai/workflow/core.py                            |      477 |       66 |      178 |       32 |     83% |149-174, 180-\>183, 184, 251, 359, 384, 462, 512-\>515, 523-\>521, 594-595, 626, 651, 672, 678-\>681, 696-697, 723, 740, 755-\>758, 771, 802, 849-\>858, 913, 953-1000, 1027-1028, 1046, 1063, 1070, 1154, 1169, 1174, 1213-1215, 1220-1221, 1229-1233, 1248-1249, 1261-1262 |
| packages/markitai/src/markitai/workflow/helpers.py                         |      222 |       19 |      112 |       15 |     89% |111, 117, 121, 148-\>150, 282-\>276, 298-\>301, 342-348, 433, 448-454, 474-475, 477-480, 487-\>486, 490-\>486, 499-\>501, 516-\>522, 520 |
| packages/markitai/src/markitai/workflow/single.py                          |      187 |        3 |       30 |        4 |     97% |149, 157, 336-\>358, 483 |
| packages/markitai/src/markitai/workflow/url.py                             |      206 |        9 |       68 |       10 |     93% |144-145, 179, 335-336, 370-372, 424, 551-\>543, 606, 607-\>618, 618-\>604, 625-\>630, 630-\>639 |
| **TOTAL**                                                                  | **29547** | **3466** | **11088** | **1585** | **86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Ynewtime/markitai/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ynewtime/markitai/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FYnewtime%2Fmarkitai%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.