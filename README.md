# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| packages/markitai/src/markitai/\_\_init\_\_.py                             |        1 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/\_\_main\_\_.py                             |        2 |        2 |        0 |        0 |      0% |       3-5 |
| packages/markitai/src/markitai/batch.py                                    |      769 |       86 |      274 |       46 |     87% |119-\>118, 121, 124, 127, 272-274, 396, 603, 605, 613-618, 679-680, 696-\>700, 705-\>exit, 729-\>732, 734-\>exit, 822, 853-854, 858-861, 878, 883, 886, 900-901, 941-942, 949, 952-954, 962-\>946, 970-982, 991, 1013, 1023-\>1029, 1031-1032, 1059-\>1065, 1081, 1092-1093, 1107-\>1085, 1125-\>exit, 1141, 1150-1151, 1183, 1289-\>1298, 1308-1310, 1314-1319, 1328-1332, 1361-1362, 1391, 1436-1437, 1445, 1450-1453, 1464, 1481, 1539, 1545-1547, 1556-1557, 1568, 1572-1575 |
| packages/markitai/src/markitai/cli/\_\_init\_\_.py                         |       19 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/commands/\_\_init\_\_.py                |       14 |        8 |        2 |        0 |     38% |     36-43 |
| packages/markitai/src/markitai/cli/commands/auth.py                        |      225 |       21 |       96 |       20 |     87% |63-65, 68, 80-81, 83, 86, 89, 92, 161, 171, 183, 219, 224, 227, 286-\>288, 338, 385, 388-\>403, 391-\>399, 461, 474, 527 |
| packages/markitai/src/markitai/cli/commands/cache.py                       |      155 |        3 |       62 |        6 |     96% |44-\>56, 56-\>exit, 68-\>73, 70-\>73, 230-231, 246-\>exit, 286 |
| packages/markitai/src/markitai/cli/commands/config.py                      |      257 |       44 |       94 |       16 |     81% |48, 71, 73, 78-79, 81, 84, 91, 137-\>139, 147, 154, 178-180, 184-185, 187-190, 194, 267-269, 270-\>exit, 285, 405, 407, 449-461, 467-471, 484-486 |
| packages/markitai/src/markitai/cli/commands/doctor.py                      |      392 |       47 |      160 |       22 |     87% |103, 126-129, 136, 137-\>142, 178-180, 214-218, 236-256, 291-292, 319-327, 349-\>359, 400, 482-\>488, 496-497, 535, 539-540, 588-589, 653-\>649, 748-764, 792-808, 820, 849, 888, 953-\>957, 957-\>961, 990-\>977, 1160-1161 |
| packages/markitai/src/markitai/cli/commands/init.py                        |      234 |       65 |       72 |       10 |     71% |75-77, 204-224, 233-241, 250-254, 263-274, 283-291, 317-320, 328-\>327, 334-348, 352-359, 362-\>373, 421-\>447, 426-\>423, 433, 435 |
| packages/markitai/src/markitai/cli/commands/serve.py                       |       69 |       16 |       22 |        4 |     76% |37-55, 69-\>exit, 71-\>76, 74-75, 141-147, 154 |
| packages/markitai/src/markitai/cli/config\_editor.py                       |      308 |      155 |      110 |       12 |     50% |66-\>72, 85-87, 94, 126, 134-\>51, 183-361, 373, 419-\>423, 451, 456-462, 474, 491-492, 497-523, 548, 554, 560, 581-582 |
| packages/markitai/src/markitai/cli/console.py                              |        3 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/framework.py                            |       72 |        2 |       26 |        3 |     95% |134, 180-\>182, 217 |
| packages/markitai/src/markitai/cli/hints.py                                |        9 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/i18n.py                                 |       29 |        2 |       10 |        0 |     95% |   192-194 |
| packages/markitai/src/markitai/cli/interactive.py                          |      244 |       83 |       84 |       23 |     63% |36-43, 52-54, 78, 109, 123, 136-145, 164-167, 170-\>179, 174-177, 187-\>196, 189-190, 202, 230-245, 251, 270-271, 274, 277, 300-364, 376-\>383, 391-418, 432, 441-442, 445, 478-479, 483, 504-511, 513, 515, 526-\>529 |
| packages/markitai/src/markitai/cli/logging\_config.py                      |      169 |       14 |       50 |        9 |     89% |37-\>exit, 109-\>exit, 111-112, 170, 182-183, 207-208, 340, 344, 436, 439, 498, 549, 564 |
| packages/markitai/src/markitai/cli/main.py                                 |      364 |       44 |      152 |       11 |     87% |16-19, 159, 170-171, 179-180, 499-501, 570-578, 588-590, 677, 714-716, 800, 873-905, 908, 1009-1010, 1015-1016, 1025 |
| packages/markitai/src/markitai/cli/processors/\_\_init\_\_.py              |       36 |        8 |       14 |        4 |     76% |79-81, 88-90, 92-94, 100-102 |
| packages/markitai/src/markitai/cli/processors/batch.py                     |      445 |       41 |      156 |       20 |     90% |299-300, 313-314, 350-\>354, 379-380, 657-\>652, 659-660, 672-\>674, 678, 725, 757, 766-\>768, 795-806, 817, 846-\>865, 892, 918, 924-929, 952, 964-\>971, 969-970, 988, 994-999, 1028-\>1064, 1052-1053, 1068, 1070-\>1101, 1090 |
| packages/markitai/src/markitai/cli/processors/file.py                      |      154 |       16 |       54 |       14 |     85% |51, 53, 100-102, 122, 179-181, 186-\>191, 211, 258-\>277, 266, 286-\>359, 306-307, 308-\>314, 326, 334, 339, 346-\>359 |
| packages/markitai/src/markitai/cli/processors/llm.py                       |      161 |        2 |       52 |        6 |     96% |159-\>156, 168-\>175, 323-\>374, 361, 369, 462-\>465 |
| packages/markitai/src/markitai/cli/processors/url.py                       |      482 |       70 |      132 |       25 |     82% |84, 183, 305-\>307, 341-342, 349, 355-363, 377-380, 432-445, 457-493, 553-555, 575, 579, 589-591, 603-614, 625-628, 642, 719, 723, 913-914, 919, 934-942, 970, 1002, 1081, 1145-1150, 1194, 1375-1389 |
| packages/markitai/src/markitai/cli/processors/validators.py                |      109 |        8 |       48 |        1 |     93% |102-110, 189-190 |
| packages/markitai/src/markitai/cli/providers\_detect.py                    |       75 |       16 |       20 |        1 |     82% |34-41, 46-53, 63-64, 148-\>159 |
| packages/markitai/src/markitai/cli/ui.py                                   |      261 |        5 |       84 |        4 |     97% |421-\>exit, 449, 452, 512-513, 599 |
| packages/markitai/src/markitai/config.py                                   |      467 |       41 |      136 |       25 |     87% |584, 586, 669, 715, 724, 836, 839, 841, 844, 896, 978-\>990, 997, 1013, 1044, 1048, 1051, 1057, 1060, 1100-1103, 1106, 1113-1115, 1124-1129, 1151, 1170-1179, 1182-1183, 1187-1191 |
| packages/markitai/src/markitai/constants.py                                |       95 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/\_\_init\_\_.py                   |       13 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/\_patches.py                      |       47 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/base.py                           |       90 |        1 |        2 |        0 |     99% |       208 |
| packages/markitai/src/markitai/converter/cloudflare.py                     |       55 |        4 |       12 |        4 |     88% |121-\>123, 124-128, 136, 166 |
| packages/markitai/src/markitai/converter/eml.py                            |      133 |       19 |       46 |       10 |     83% |65, 68, 77-79, 85, 103-119, 129-\>124, 139, 146-148, 187-188, 219-\>222, 265-\>270, 267-\>270 |
| packages/markitai/src/markitai/converter/heif.py                           |       28 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/image.py                          |       83 |        8 |       26 |        3 |     90% |80-81, 173-179, 223-224 |
| packages/markitai/src/markitai/converter/kreuzberg.py                      |       33 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/legacy.py                         |      219 |       96 |       76 |       11 |     52% |121-124, 161-192, 211-216, 259-299, 317-332, 357-394, 431, 436, 452, 455-\>458, 459-\>450, 502-506, 525, 553, 587-592, 599, 620 |
| packages/markitai/src/markitai/converter/markitdown\_ext.py                |       70 |        3 |        8 |        0 |     96% |68-69, 193 |
| packages/markitai/src/markitai/converter/office.py                         |      221 |       80 |       52 |        9 |     62% |67, 134-151, 182-\>186, 192-\>199, 227-246, 252-348, 371-375, 391-395, 411-412, 439-446, 450-451, 539-\>543 |
| packages/markitai/src/markitai/converter/pdf.py                            |      558 |       47 |      194 |       25 |     90% |116-\>112, 135, 145, 166, 171-172, 269, 307-309, 365-368, 377, 414, 418, 528, 559-\>647, 563-\>561, 614-615, 671-\>674, 748, 768, 844, 909-911, 972-\>970, 1015-1016, 1035-\>1039, 1057-1059, 1117-1119, 1156-1158, 1168-\>1170, 1170-\>1172, 1187-1189, 1205-1207, 1252-1255, 1259-\>1263, 1289-\>1301, 1303, 1306 |
| packages/markitai/src/markitai/converter/text.py                           |       15 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/webextract\_html\_converter.py    |        2 |        2 |        0 |        0 |      0% |      8-11 |
| packages/markitai/src/markitai/domain\_profiles.py                         |        4 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/fetch.py                                    |      387 |       41 |      170 |       21 |     87% |249, 385-386, 425, 453, 456-\>458, 500-\>508, 520, 535, 598-600, 607, 612, 644-\>646, 679-\>681, 714-733, 739, 743-\>746, 873-874, 887, 1027, 1063, 1067-\>1069, 1074-1077, 1086-1088, 1106-1108, 1121-1123 |
| packages/markitai/src/markitai/fetch\_cache.py                             |      272 |       12 |       56 |        3 |     95% |39, 44, 454-463, 562-563, 621-622 |
| packages/markitai/src/markitai/fetch\_consent.py                           |      107 |        4 |       38 |        4 |     94% |72, 140, 158, 190 |
| packages/markitai/src/markitai/fetch\_http.py                              |      157 |       50 |       38 |        6 |     65% |57-58, 64-69, 86, 90, 120-121, 133-137, 145, 183-184, 188, 216-245, 255-257, 273-285, 306-\>308, 310 |
| packages/markitai/src/markitai/fetch\_playwright.py                        |      420 |       75 |      158 |       26 |     78% |104, 109, 121-168, 279-280, 407, 452-456, 461-468, 558, 580, 601-602, 607, 612-\>621, 617-618, 624-625, 631-632, 644-\>686, 672, 675-\>678, 693-699, 700-\>738, 702-\>738, 729-736, 777-780, 836-838, 880, 882, 884, 886, 888, 890, 892, 894, 896, 1054-1055 |
| packages/markitai/src/markitai/fetch\_policy.py                            |      244 |       28 |      122 |       16 |     88% |50, 59, 73, 93, 185, 196, 219, 223, 225, 253-254, 257, 272-273, 298-303, 311-312, 325-326, 329, 341, 346-347, 350, 380, 385 |
| packages/markitai/src/markitai/fetch\_screenshot.py                        |       55 |        7 |       14 |        1 |     88% |70-71, 74-77, 128-129 |
| packages/markitai/src/markitai/fetch\_session.py                           |      280 |       89 |      110 |       10 |     62% |53-\>40, 86-127, 130-192, 325-\>327, 341-342, 345-\>370, 353-359, 367, 449-454, 462, 514-516, 531-532, 548-551, 557-558, 605 |
| packages/markitai/src/markitai/fetch\_strategies/\_\_init\_\_.py           |       34 |        0 |        6 |        2 |     95% |70-\>exit, 72-\>exit |
| packages/markitai/src/markitai/fetch\_strategies/\_shared.py               |       45 |        2 |        8 |        3 |     91% |57, 84, 94-\>97 |
| packages/markitai/src/markitai/fetch\_strategies/cloudflare.py             |      106 |        9 |       42 |       10 |     87% |94-\>96, 96-\>98, 98-\>104, 149-\>193, 194, 215-216, 221, 243, 246, 260, 270-275 |
| packages/markitai/src/markitai/fetch\_strategies/defuddle.py               |       64 |       19 |       20 |        5 |     62% |82, 94, 107-117, 121-124, 127, 139-142 |
| packages/markitai/src/markitai/fetch\_strategies/jina.py                   |       84 |        7 |       30 |        8 |     87% |40-\>38, 42, 112, 114, 116, 141, 162-\>167, 185-186 |
| packages/markitai/src/markitai/fetch\_strategies/playwright.py             |       34 |        2 |       14 |        2 |     92% |    31, 44 |
| packages/markitai/src/markitai/fetch\_strategies/static.py                 |      134 |       20 |       62 |        8 |     82% |47-\>53, 53-\>60, 74-\>81, 85-86, 94-103, 121, 278, 334-347 |
| packages/markitai/src/markitai/fetch\_support.py                           |       41 |        2 |       16 |        1 |     95% |   118-119 |
| packages/markitai/src/markitai/fetch\_types.py                             |       39 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/image.py                                    |      634 |       80 |      188 |       22 |     86% |110-\>117, 134, 138-140, 192, 249-253, 315-332, 347-382, 407, 466, 482-484, 657, 714, 721-728, 737-\>740, 746, 757-764, 855, 955-957, 1075-1077, 1118-1119, 1136, 1195, 1204, 1277-1279, 1308-1309, 1326, 1415, 1513-\>1545, 1541-1542 |
| packages/markitai/src/markitai/json\_order.py                              |      175 |       24 |      104 |       11 |     82% |226, 231-\>239, 317-\>321, 384-\>388, 398-\>404, 411-423, 448-\>467, 456-463, 467-\>483, 473-480, 502-\>519 |
| packages/markitai/src/markitai/llm/\_\_init\_\_.py                         |        7 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/cache.py                                |      254 |        8 |       74 |        4 |     96% |181, 407-408, 412-413, 502-\>509, 526-\>exit, 529-530, 540 |
| packages/markitai/src/markitai/llm/content.py                              |      303 |       29 |      154 |       16 |     88% |80, 95-\>91, 297, 306-\>309, 315-344, 355-\>353, 357-\>364, 370-\>369, 373-\>369, 377-\>369, 385-389, 420-\>456, 584-586, 587-\>580, 591-592, 658 |
| packages/markitai/src/markitai/llm/degeneration.py                         |       68 |        0 |       32 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/document.py                             |      547 |       28 |      126 |       10 |     94% |176-177, 338, 428, 460-461, 482, 485, 498, 504, 557, 1103-1107, 1165-\>1196, 1482-1513, 1799-\>1801 |
| packages/markitai/src/markitai/llm/engine.py                               |      281 |       18 |       96 |       11 |     92% |98-100, 105, 111-115, 150, 191-\>189, 287, 432-\>439, 443-\>451, 699, 753-761, 790-796, 828 |
| packages/markitai/src/markitai/llm/models.py                               |       73 |        0 |       22 |        2 |     98% |101-\>103, 144-\>148 |
| packages/markitai/src/markitai/llm/processor.py                            |      747 |      140 |      272 |       34 |     79% |36-37, 185-194, 236, 268-\>282, 288-302, 458, 462-471, 512, 535-538, 555-561, 722, 825-\>834, 1023, 1057-1058, 1088-1094, 1114, 1133-1139, 1168, 1173, 1193-1203, 1243, 1247-1248, 1273-1279, 1304-1309, 1315, 1332, 1337, 1351-1357, 1460-1468, 1520-1523, 1541-\>1539, 1543-\>1545, 1545-\>1539, 1551-\>1557, 1620-1622, 1762, 1783-\>1789, 1786-\>1789, 1790, 1827-1845, 1862-1863, 1872-1929, 1948-\>1952 |
| packages/markitai/src/markitai/llm/types.py                                |       81 |        5 |       16 |        5 |     90% |156, 159, 169, 171, 176 |
| packages/markitai/src/markitai/llm/vision.py                               |      386 |       18 |      108 |       15 |     93% |97, 105, 163, 267, 526-528, 709, 712-\>720, 724-\>735, 818, 820, 835, 837, 844-848, 946-\>951, 1154-\>1188, 1168-\>1170, 1181-1182, 1188-\>1222, 1202-\>1204, 1215-1216 |
| packages/markitai/src/markitai/ocr.py                                      |      259 |       53 |       76 |       16 |     76% |93-\>95, 96-\>98, 103-\>105, 105-\>exit, 165-\>175, 219-220, 232-\>251, 317-319, 332-339, 353, 355, 366, 395, 425-426, 432, 458, 463, 476-491, 510-534, 558-569, 586-587, 640-642 |
| packages/markitai/src/markitai/ports.py                                    |       27 |        3 |        2 |        0 |     90% |22, 26, 32 |
| packages/markitai/src/markitai/prompts/\_\_init\_\_.py                     |       64 |       12 |       32 |        4 |     77% |100-\>104, 116, 131-\>134, 157-169 |
| packages/markitai/src/markitai/providers/\_\_init\_\_.py                   |      294 |       44 |      118 |       15 |     83% |144, 156-157, 196, 233-235, 282-\>294, 301-\>313, 309-310, 318-319, 403-409, 418-424, 462-468, 483-489, 501-504, 506-\>512, 557, 560, 588-591, 635, 646-648, 682 |
| packages/markitai/src/markitai/providers/auth.py                           |      343 |       88 |      110 |        6 |     74% |52-53, 267-275, 285-302, 311-315, 336, 357-388, 393-428, 433-440, 516-\>520, 559-560, 635-639, 647-648, 847, 864-872, 913-914, 925 |
| packages/markitai/src/markitai/providers/chatgpt.py                        |      174 |       40 |       64 |       12 |     71% |55-57, 69-71, 88, 117-118, 127, 175-176, 195, 198, 201-202, 209-\>193, 219-223, 236-237, 269-272, 281, 287-\>279, 295, 336-\>345, 351, 363-364, 369-370, 410, 481-487 |
| packages/markitai/src/markitai/providers/claude\_agent.py                  |      178 |       22 |       76 |       16 |     85% |74-78, 100, 162, 167-\>160, 186-190, 222-227, 231-236, 262-\>exit, 305-\>308, 318, 351, 354, 380, 387-\>375, 457-\>456, 459-\>454, 461-\>465, 486-489, 573 |
| packages/markitai/src/markitai/providers/common.py                         |       37 |        8 |       22 |        1 |     81% |27-\>24, 100-115 |
| packages/markitai/src/markitai/providers/copilot.py                        |      321 |      110 |      102 |       26 |     62% |81-82, 94-98, 114-174, 211, 231-232, 236-273, 304, 306, 311-\>310, 315-\>310, 352-361, 365, 367, 375-\>373, 377-378, 401-405, 417, 461, 465-466, 491, 493-\>533, 502, 567, 570, 579, 604-605, 621-624, 650, 662, 666-670, 673-674, 680, 702-705, 711-712, 731-733, 771-\>781, 805, 810-\>817, 813-814 |
| packages/markitai/src/markitai/providers/discovery.py                      |      321 |       48 |      136 |       25 |     82% |49-53, 90, 179-180, 261, 327, 343-387, 398, 416, 420-\>422, 423-\>425, 440, 459-463, 468, 498-\>574, 505-\>511, 516-\>518, 518-\>574, 525-\>527, 545, 549-\>574, 562-\>574, 586, 591-\>584, 599-\>597, 607-\>603, 620, 661, 671-673, 706 |
| packages/markitai/src/markitai/providers/errors.py                         |       43 |        1 |        8 |        2 |     94% |284, 305-\>exit |
| packages/markitai/src/markitai/providers/json\_mode.py                     |       59 |        7 |       26 |        6 |     85% |109-110, 173-\>188, 175, 179, 181, 183, 185 |
| packages/markitai/src/markitai/providers/oauth\_display.py                 |       53 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/providers/timeout.py                        |       48 |        2 |       26 |        5 |     91% |138, 143-\>135, 147, 152-\>145, 154-\>145 |
| packages/markitai/src/markitai/runs/\_\_init\_\_.py                        |        4 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/runs/output.py                              |       82 |        6 |       22 |        2 |     92% |139-140, 226-\>234, 231-232, 244-245 |
| packages/markitai/src/markitai/runs/report.py                              |       32 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/runs/types.py                               |       22 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/security.py                                 |      157 |       33 |       50 |        8 |     74% |38-50, 68-80, 125-\>131, 128-129, 137, 196, 243, 262-263, 353, 368, 371-372 |
| packages/markitai/src/markitai/serve/\_\_init\_\_.py                       |       14 |        2 |        2 |        0 |     88% |     64-65 |
| packages/markitai/src/markitai/serve/app.py                                |     1180 |      107 |      426 |       77 |     88% |149-155, 174-179, 197-\>211, 222-223, 233-234, 241, 253, 303-\>349, 382, 412-413, 444-\>450, 559, 584, 587, 590-591, 627, 635, 697, 754, 779-780, 821, 824, 844, 847, 885, 982-988, 995, 1035, 1070, 1091-1094, 1101, 1111, 1128, 1131, 1148-1149, 1165, 1167, 1206-1207, 1256, 1294-1295, 1314-1315, 1447, 1510-1528, 1554, 1569, 1621-1626, 1631-1636, 1669, 1677, 1725, 1739, 1741, 1753, 1855, 1862, 1888, 1914, 1932, 1945-\>1947, 1966-1972, 1974-\>1979, 1976, 1979-\>1985, 1981, 1988, 1991, 2003, 2004-\>2009, 2006, 2009-\>2014, 2011, 2035, 2047-\>2049, 2068-\>2073, 2076-2077, 2084, 2091, 2132-2133, 2364, 2377-\>2386, 2379-\>2386, 2382-2385, 2419-2421, 2441, 2448, 2460-2463, 2469-\>exit, 2606-\>2615 |
| packages/markitai/src/markitai/serve/jobs.py                               |      479 |      100 |      112 |       22 |     77% |247-\>exit, 272, 274-277, 287, 299-300, 379, 384, 397, 447-450, 461-464, 475-476, 478, 482-483, 532, 535-536, 545-547, 560-561, 580-581, 638, 646-655, 673-696, 699-\>711, 737-738, 785-790, 859-873, 899-909, 917-918, 961, 963-\>exit, 976-\>994, 988-\>994, 990-992, 998-1010 |
| packages/markitai/src/markitai/serve/schemas.py                            |      130 |        4 |       20 |        4 |     95% |31, 70, 135, 207 |
| packages/markitai/src/markitai/types.py                                    |        8 |        8 |        0 |        0 |      0% |      3-19 |
| packages/markitai/src/markitai/urls.py                                     |       75 |       12 |       36 |        7 |     83% |84-85, 88, 98, 100-101, 114-115, 117-118, 127, 160 |
| packages/markitai/src/markitai/utils/\_\_init\_\_.py                       |        9 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/asset\_store.py                       |       47 |        1 |       14 |        3 |     93% |29, 84-\>90, 87-\>84 |
| packages/markitai/src/markitai/utils/cli\_helpers.py                       |       54 |        3 |       14 |        3 |     91% |50-51, 59-\>70, 73, 125-\>139 |
| packages/markitai/src/markitai/utils/executor.py                           |       82 |       25 |       28 |        8 |     66% |49-\>54, 75-97, 103-\>102, 107-119, 130, 134, 136, 140 |
| packages/markitai/src/markitai/utils/frontmatter.py                        |      171 |       10 |       84 |       10 |     92% |32, 93, 106, 119-122, 239, 243, 251, 256, 334-\>341, 360-\>352 |
| packages/markitai/src/markitai/utils/guidance.py                           |       28 |        0 |        8 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/markdown\_quality.py                  |        9 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/mime.py                               |       42 |        3 |       18 |        3 |     90% |132, 157, 162 |
| packages/markitai/src/markitai/utils/office.py                             |      116 |       68 |       34 |        5 |     35% |50-56, 78-97, 135-168, 183-\>181, 188-225, 238, 264, 270-272 |
| packages/markitai/src/markitai/utils/office\_mac.py                        |      201 |       11 |       54 |        3 |     95% |414, 417, 439-\>441, 454-455, 463-465, 474-475, 483-484 |
| packages/markitai/src/markitai/utils/output.py                             |       26 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/paths.py                              |       31 |        2 |        4 |        0 |     89% |     20-21 |
| packages/markitai/src/markitai/utils/term.py                               |       48 |        1 |       18 |        2 |     95% |52, 84-\>86 |
| packages/markitai/src/markitai/utils/terminal\_image.py                    |       66 |        1 |       22 |        1 |     98% |        83 |
| packages/markitai/src/markitai/utils/text.py                               |      261 |       42 |      118 |       15 |     82% |82-87, 92, 100-120, 152, 181, 214, 229, 304-\>311, 327, 509, 528-535, 569, 621-622, 645-647, 649-\>611, 654 |
| packages/markitai/src/markitai/utils/url\_redaction.py                     |       27 |        5 |        4 |        1 |     81% |20, 25-26, 33-34 |
| packages/markitai/src/markitai/webextract/\_\_init\_\_.py                  |       31 |        3 |       10 |        2 |     88% |32, 59, 89 |
| packages/markitai/src/markitai/webextract/constants.py                     |       19 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/content\_boundary.py             |      132 |       14 |       92 |       12 |     88% |54, 58, 69, 73, 88, 144-147, 149, 170, 173-\>172, 181, 192, 194 |
| packages/markitai/src/markitai/webextract/dom.py                           |        8 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/\_\_init\_\_.py         |        5 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/callouts.py             |       57 |        1 |       26 |        5 |     93% |34-\>41, 42-\>46, 54-\>61, 74, 77-\>83 |
| packages/markitai/src/markitai/webextract/elements/code.py                 |       44 |        2 |       28 |        2 |     94% |    77, 81 |
| packages/markitai/src/markitai/webextract/elements/footnotes.py            |     1018 |      313 |      564 |       94 |     65% |72-73, 79-80, 123, 136, 146, 148, 155, 162, 182, 186-192, 201-\>197, 209, 213, 277, 291-\>290, 308, 314-315, 328-331, 346, 353, 387-\>386, 414, 417, 421-\>423, 433-442, 448-461, 465-482, 488-504, 510-\>516, 535-551, 573, 596, 598-\>594, 602, 606, 615-\>619, 624, 627-\>632, 655-688, 694-696, 701-744, 779-781, 784, 802-833, 840-\>exit, 859-\>868, 861-864, 880-883, 899, 901-\>895, 903-\>895, 958-\>965, 974, 977, 1002, 1009, 1037, 1048-1049, 1082-\>1079, 1087, 1096-\>1118, 1102-\>1115, 1107-1112, 1114, 1143-1196, 1200, 1209, 1233-\>1240, 1243-1276, 1286-1316, 1330-1359, 1381, 1383, 1392, 1395, 1398, 1403-\>1411, 1421-\>1424, 1425, 1441, 1472, 1474, 1495, 1498, 1501, 1509, 1511, 1514, 1543-\>1542, 1558-\>1563, 1564, 1571-\>1576, 1577, 1584, 1591-1594, 1625-1631 |
| packages/markitai/src/markitai/webextract/elements/headings.py             |       19 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/images.py               |       57 |        2 |       26 |        4 |     93% |28-\>31, 72, 90, 91-\>69 |
| packages/markitai/src/markitai/webextract/elements/math.py                 |       69 |        1 |       36 |        5 |     94% |68-\>57, 91-\>93, 97-\>87, 99-\>87, 129 |
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
| packages/markitai/src/markitai/webextract/extractors/registry.py           |       17 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/extractors/steam\_news.py        |       83 |       10 |       20 |        7 |     83% |40-41, 71-\>73, 73-\>75, 75-\>80, 100, 105-107, 114, 119-121 |
| packages/markitai/src/markitai/webextract/extractors/x\_article.py         |       13 |        1 |        2 |        1 |     87% |        44 |
| packages/markitai/src/markitai/webextract/extractors/x\_common.py          |      356 |       44 |      220 |       56 |     82% |87, 91, 93, 146, 149-\>143, 155-\>159, 157-\>159, 159-\>164, 185, 188, 201, 221, 225-\>219, 244-\>246, 279, 283-285, 290-\>292, 294, 306, 312, 317, 342, 345, 347, 351, 358, 395, 420, 448, 501, 507, 509-\>504, 511-\>504, 513-\>499, 521, 530-\>519, 533-\>519, 565, 569, 578-579, 584-\>581, 589, 623-\>611, 628, 645, 646-\>643, 666, 671, 672-\>675, 700, 703, 706, 710, 726-\>730, 729, 773-775 |
| packages/markitai/src/markitai/webextract/extractors/x\_tweet.py           |      132 |       14 |       72 |       17 |     84% |63-\>65, 66, 71, 116, 129-\>152, 138, 147-\>136, 150, 171-\>174, 174-\>182, 180-181, 185-\>188, 204, 219-220, 247, 261-262, 267-\>265, 276 |
| packages/markitai/src/markitai/webextract/extractors/youtube\_page.py      |       97 |       29 |       56 |       17 |     58% |57-58, 83-\>85, 85-\>87, 87-\>90, 122-\>126, 135-151, 172-179, 199-210, 232-\>235, 235-\>238, 240-\>251, 243-\>251 |
| packages/markitai/src/markitai/webextract/frontmatter.py                   |       14 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/html\_to\_markdown.py            |      276 |       23 |      156 |       27 |     88% |57, 62, 174, 177, 187, 209, 220, 228, 237, 245, 257, 278, 286, 295, 327-\>324, 330, 347, 372, 374, 381, 384, 392, 417-\>419, 422-\>430, 428-\>430, 436, 494 |
| packages/markitai/src/markitai/webextract/markdown.py                      |      135 |       13 |       56 |       12 |     87% |89, 94-\>96, 114, 121-124, 177, 180, 222, 230, 238, 270, 276, 292 |
| packages/markitai/src/markitai/webextract/metadata.py                      |      163 |       10 |      102 |       13 |     91% |60, 115, 172, 176-\>179, 181, 205, 238, 241-242, 254, 257-\>256, 261-\>exit, 263-\>262, 279, 282-\>276 |
| packages/markitai/src/markitai/webextract/mobile\_styles.py                |       58 |        5 |       34 |        5 |     89% |28, 38-\>37, 41-42, 68, 72, 81-\>74 |
| packages/markitai/src/markitai/webextract/pipeline.py                      |      191 |       11 |       54 |       11 |     90% |107-\>106, 116, 139-140, 299-\>308, 309-\>320, 320-\>322, 401-403, 405, 431-\>448, 474-476, 478, 510-\>513 |
| packages/markitai/src/markitai/webextract/preprocess.py                    |       54 |        1 |       14 |        1 |     97% |       133 |
| packages/markitai/src/markitai/webextract/quality.py                       |      124 |        3 |       48 |        2 |     97% |87, 135-136 |
| packages/markitai/src/markitai/webextract/removals/\_\_init\_\_.py         |       19 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/removals/content\_patterns.py    |      867 |      183 |      484 |       75 |     77% |162, 165, 230-231, 275, 277, 279, 284-285, 306-\>305, 313, 343, 344-\>346, 347, 380, 405, 418, 423-431, 440, 442, 468-469, 483, 490, 502-503, 507, 520, 528-536, 547, 552-\>558, 554-555, 562-563, 565-566, 611-616, 620-622, 652-\>659, 666-672, 682-686, 699-\>706, 718-726, 734-\>593, 752, 761-762, 766-768, 772, 788, 811-\>817, 818-\>823, 820-\>818, 825-830, 838-839, 868-\>870, 875-876, 897-898, 906, 912, 917-\>927, 923, 931-960, 969, 980-\>975, 991-1003, 1019-\>1035, 1025-1026, 1069-1071, 1080-1081, 1093, 1107, 1118, 1123, 1125, 1136, 1157-1166, 1174, 1181-1189, 1197, 1208, 1226-1235, 1244, 1248-1260, 1275-\>1279 |
| packages/markitai/src/markitai/webextract/removals/hidden.py               |       61 |        8 |       40 |        8 |     84% |49, 97, 105, 109, 111, 116, 118, 122 |
| packages/markitai/src/markitai/webextract/removals/scoring.py              |      108 |       10 |       66 |        5 |     90% |76, 78-79, 87, 107-108, 128, 163-\>169, 188-190 |
| packages/markitai/src/markitai/webextract/removals/selectors.py            |      120 |       22 |       82 |        8 |     81% |46, 51-64, 84, 87, 101-\>105, 116, 130-132, 159, 164 |
| packages/markitai/src/markitai/webextract/removals/small\_images.py        |       33 |        2 |       16 |        0 |     96% |     57-58 |
| packages/markitai/src/markitai/webextract/render.py                        |      128 |        3 |       64 |        6 |     95% |106, 159, 162, 188-\>194, 213-\>215, 215-\>217 |
| packages/markitai/src/markitai/webextract/resolver.py                      |       31 |        0 |        8 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/sanitize.py                      |       26 |        0 |       18 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/schema.py                        |       38 |        4 |       20 |        2 |     90% |23, 26-27, 43 |
| packages/markitai/src/markitai/webextract/scoring.py                       |      139 |       15 |       70 |        5 |     89% |77, 87, 100, 171-172, 187-194, 217, 231 |
| packages/markitai/src/markitai/webextract/semantics.py                     |       38 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/standardize.py                   |      169 |        7 |      104 |        8 |     95% |173, 196, 226, 261, 263, 318, 341, 369-\>367 |
| packages/markitai/src/markitai/webextract/thread\_policy.py                |       16 |        1 |        4 |        1 |     90% |        62 |
| packages/markitai/src/markitai/webextract/types.py                         |       44 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/utils.py                         |       16 |        0 |        2 |        0 |    100% |           |
| packages/markitai/src/markitai/workflow/\_\_init\_\_.py                    |       15 |        4 |        6 |        2 |     71% |30-32, 34-36 |
| packages/markitai/src/markitai/workflow/core.py                            |      494 |       71 |      188 |       38 |     82% |161-\>166, 177-202, 212, 279, 381, 406, 484, 486, 528-\>531, 539-\>537, 611, 619-620, 631, 640, 661, 665, 686, 692-\>695, 709-710, 736, 753, 768-\>771, 783, 814, 861-\>870, 924, 964-1011, 1038-1039, 1056, 1073, 1080, 1163, 1173, 1178, 1183, 1222-1224, 1229-1230, 1238-1242, 1257-1258, 1270-1271 |
| packages/markitai/src/markitai/workflow/helpers.py                         |      207 |       16 |      106 |       16 |     90% |111, 117, 121, 148-\>150, 151, 279-\>273, 295-\>298, 335-338, 407, 417, 437-438, 440-443, 450-\>449, 453-\>449, 462-\>464, 479-\>485, 483 |
| packages/markitai/src/markitai/workflow/single.py                          |      191 |        4 |       34 |        6 |     96% |42, 149, 157, 336-\>358, 483 |
| **TOTAL**                                                                  | **25507** | **3561** | **9614** | **1444** | **83%** |           |


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