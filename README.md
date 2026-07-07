# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Ynewtime/markitai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                       |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| packages/markitai/src/markitai/\_\_init\_\_.py                             |        1 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/\_\_main\_\_.py                             |        2 |        2 |        0 |        0 |      0% |       3-5 |
| packages/markitai/src/markitai/batch.py                                    |      780 |       98 |      278 |       52 |     85% |117-\>116, 119, 122, 125, 268-270, 351, 391, 590, 592, 600-605, 666-667, 678-\>683, 698-\>exit, 719-\>exit, 722-\>725, 727-\>exit, 740-\>exit, 755-\>exit, 815, 846-847, 851-854, 871, 876, 879, 893-894, 934-935, 942, 945-947, 955-\>939, 963-975, 984, 1006, 1016-\>1022, 1024-1025, 1052-\>1058, 1074, 1085-1086, 1100-\>1078, 1108-1109, 1118-\>exit, 1134, 1143-1144, 1176, 1199-1209, 1307-\>1316, 1326-1328, 1332-1337, 1346-1350, 1379-1380, 1409, 1454-1455, 1463, 1468-1471, 1480-1490, 1505, 1558, 1564-1566, 1575-1576, 1587, 1591-1594 |
| packages/markitai/src/markitai/cli/\_\_init\_\_.py                         |       20 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/commands/\_\_init\_\_.py                |       14 |        8 |        2 |        0 |     38% |     34-41 |
| packages/markitai/src/markitai/cli/commands/auth.py                        |      268 |       31 |      110 |       22 |     84% |72-74, 77, 89-90, 92, 95, 98, 101, 170, 180, 194, 206, 242, 247, 250, 359, 404-409, 452-453, 462, 512, 513-\>528, 516-\>524, 586, 599, 652 |
| packages/markitai/src/markitai/cli/commands/cache.py                       |      155 |        3 |       62 |        5 |     96% |44-\>56, 56-\>exit, 68-\>73, 70-\>73, 230-231, 286 |
| packages/markitai/src/markitai/cli/commands/config.py                      |      198 |       41 |       62 |       11 |     78% |47-\>49, 57, 64, 88-90, 94-95, 97-100, 104, 159-172, 173-\>exit, 188, 308, 310, 352-364, 370-374, 387-389 |
| packages/markitai/src/markitai/cli/commands/doctor.py                      |      311 |       51 |      118 |       20 |     83% |120-122, 135-138, 151-157, 175-195, 230-231, 258-266, 285-\>287, 287-\>289, 296-304, 399-\>405, 413-414, 452, 456-457, 505-506, 635-651, 679-695, 707, 739, 817-\>821, 821-\>825, 868-876, 959-960 |
| packages/markitai/src/markitai/cli/commands/init.py                        |      243 |       66 |       76 |       11 |     71% |75-77, 203-223, 232-240, 249-253, 262-273, 282-290, 315-318, 326-\>325, 332-346, 350-357, 360-\>371, 446-\>474, 451-\>448, 458, 460, 462 |
| packages/markitai/src/markitai/cli/config\_editor.py                       |      308 |      155 |      110 |       12 |     50% |66-\>72, 85-87, 94, 126, 134-\>51, 183-361, 373, 419-\>423, 451, 456-462, 474, 491-492, 497-523, 548, 554, 560, 581-582 |
| packages/markitai/src/markitai/cli/console.py                              |       12 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/framework.py                            |       65 |        2 |       26 |        3 |     95% |113, 159-\>161, 196 |
| packages/markitai/src/markitai/cli/hints.py                                |        9 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/cli/i18n.py                                 |       29 |        2 |       10 |        0 |     95% |   153-155 |
| packages/markitai/src/markitai/cli/interactive.py                          |      244 |       83 |       84 |       23 |     63% |36-43, 52-54, 78, 109, 123, 136-145, 164-167, 170-\>179, 174-177, 187-\>196, 189-190, 202, 230-245, 251, 270-271, 274, 277, 300-364, 376-\>383, 391-418, 432, 441-442, 445, 478-479, 483, 504-511, 513, 515, 526-\>529 |
| packages/markitai/src/markitai/cli/logging\_config.py                      |      147 |       14 |       44 |        8 |     88% |101-\>exit, 103-104, 146, 158-159, 188-189, 288, 292, 384, 387, 427, 455, 468 |
| packages/markitai/src/markitai/cli/main.py                                 |      360 |       44 |      152 |       11 |     87% |16-19, 158, 169-170, 172-173, 490-492, 559-567, 577-579, 666, 703-705, 784, 863-897, 900, 1001-1002, 1007-1008, 1017 |
| packages/markitai/src/markitai/cli/output\_manager.py                      |       59 |        1 |       12 |        1 |     97% |       129 |
| packages/markitai/src/markitai/cli/processors/\_\_init\_\_.py              |       40 |        8 |       16 |        4 |     79% |94-96, 103-105, 107-109, 115-117 |
| packages/markitai/src/markitai/cli/processors/batch.py                     |      425 |       60 |      126 |       22 |     85% |163-167, 213-\>217, 284-285, 295-296, 342-343, 445-449, 505-509, 634-\>629, 636-637, 649, 680, 688-\>690, 715-726, 737, 778-\>797, 824, 849-860, 881, 893-\>900, 898-899, 917-928, 944-\>992, 956-\>992, 980-981, 996, 998-\>1028, 1017, 1032 |
| packages/markitai/src/markitai/cli/processors/file.py                      |      204 |       32 |       74 |       16 |     81% |116-\>124, 121-122, 132-133, 144-151, 198-201, 206-\>210, 225, 290-292, 298-\>306, 326, 399-\>422, 407, 418-419, 427-\>501, 447-448, 449-\>455, 467, 475, 480, 487-\>501, 491-497, 503 |
| packages/markitai/src/markitai/cli/processors/llm.py                       |      155 |        2 |       52 |        6 |     96% |144-\>141, 153-\>160, 306-\>357, 344, 352, 445-\>448 |
| packages/markitai/src/markitai/cli/processors/url.py                       |      519 |       86 |      138 |       25 |     81% |68, 162, 319-320, 325, 331-339, 353-356, 403-431, 443-481, 539-545, 566, 570, 586-588, 600-611, 622-631, 645, 742-743, 748, 752, 956-962, 973-979, 993-\>995, 1009-1019, 1028-1030, 1105-1109, 1180, 1246, 1308-1313, 1357 |
| packages/markitai/src/markitai/cli/processors/validators.py                |      109 |        8 |       48 |        1 |     93% |102-110, 189-190 |
| packages/markitai/src/markitai/cli/providers\_detect.py                    |       81 |       18 |       22 |        2 |     81% |33-40, 45-52, 62-63, 73-74, 142-\>153, 154-\>165 |
| packages/markitai/src/markitai/cli/ui.py                                   |      137 |        3 |       44 |        3 |     97% |49, 81-\>83, 364-365, 374-\>exit |
| packages/markitai/src/markitai/config.py                                   |      457 |       45 |      136 |       23 |     85% |555, 557, 640, 685-687, 694-696, 798, 801, 803, 806, 856, 938-\>950, 957, 973, 1004, 1008, 1011, 1017, 1020, 1060-1063, 1066, 1073-1075, 1084-1089, 1111, 1130-1139, 1142-1143, 1147-1151 |
| packages/markitai/src/markitai/constants.py                                |       95 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/\_\_init\_\_.py                   |       13 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/\_patches.py                      |       47 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/base.py                           |       90 |        1 |        2 |        0 |     99% |       207 |
| packages/markitai/src/markitai/converter/cloudflare.py                     |       55 |        4 |       12 |        4 |     88% |121-\>123, 124-128, 136, 166 |
| packages/markitai/src/markitai/converter/eml.py                            |      133 |       19 |       46 |       10 |     83% |65, 68, 77-79, 85, 103-119, 129-\>124, 139, 146-148, 187-188, 219-\>222, 265-\>270, 267-\>270 |
| packages/markitai/src/markitai/converter/heif.py                           |       28 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/image.py                          |       82 |        8 |       26 |        3 |     90% |79-80, 172-178, 218-219 |
| packages/markitai/src/markitai/converter/kreuzberg.py                      |       33 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/legacy.py                         |      214 |       98 |       70 |       11 |     48% |133-136, 173-204, 223-228, 271-311, 329-344, 369-406, 443, 448, 464, 467-\>470, 471-\>462, 513-517, 525-531, 544, 578-583, 590, 611 |
| packages/markitai/src/markitai/converter/markitdown\_ext.py                |       70 |        2 |        8 |        0 |     97% |     68-69 |
| packages/markitai/src/markitai/converter/office.py                         |      197 |       79 |       42 |        7 |     57% |66, 132-149, 180-\>184, 190-\>197, 225-244, 250-346, 359-371, 402-409, 413-414, 502-\>506 |
| packages/markitai/src/markitai/converter/pdf.py                            |      544 |       48 |      186 |       23 |     90% |114-\>110, 133, 143, 164, 169-170, 267, 305-307, 363-366, 375, 412, 416, 526, 553-\>625, 592-593, 604, 649-\>652, 726, 746, 822, 887-889, 978-979, 998-\>1002, 1020-1022, 1080-1082, 1119-1121, 1131-\>1133, 1133-\>1135, 1150-1152, 1168-1170, 1215-1218, 1222-\>1226, 1250-\>1262, 1264, 1267 |
| packages/markitai/src/markitai/converter/text.py                           |       15 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/converter/webextract\_html\_converter.py    |      276 |       23 |      156 |       27 |     88% |57, 62, 174, 177, 187, 209, 220, 228, 237, 245, 257, 278, 286, 295, 327-\>324, 330, 347, 372, 374, 381, 384, 392, 417-\>419, 422-\>430, 428-\>430, 436, 494 |
| packages/markitai/src/markitai/domain\_profiles.py                         |        4 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/fetch.py                                    |     1098 |      221 |      486 |       61 |     77% |202, 326-327, 428-469, 472-534, 587-589, 604-605, 631, 698-\>685, 721-723, 729-741, 788, 793, 800, 813-823, 827-830, 833, 843-848, 866-\>864, 868, 950-953, 957-961, 1013-1017, 1025, 1163-1164, 1218-1219, 1222-1225, 1276-1277, 1295, 1312-\>1318, 1318-\>1325, 1339-\>1346, 1350-1351, 1359-1368, 1381, 1398-1399, 1408, 1418-\>1421, 1592, 1702-\>1704, 1704-\>1706, 1706-\>1712, 1802, 1856, 1858, 1860, 1885, 1906-\>1911, 1929-1930, 1944, 1962, 2054, 2061-2065, 2072-\>2080, 2090, 2150-2152, 2159, 2164, 2196-\>2198, 2231-\>2233, 2266-2285, 2291, 2295-\>2298, 2425-2426, 2439, 2603-2605, 2619-2621, 2642-2644, 2703-2705, 2718-2763, 2765-\>2594, 2779-2781, 2794-2796, 2801-2804 |
| packages/markitai/src/markitai/fetch\_cache.py                             |      270 |       12 |       56 |        3 |     95% |35, 40, 449-458, 557-558, 616-617 |
| packages/markitai/src/markitai/fetch\_http.py                              |      157 |       50 |       38 |        6 |     65% |57-58, 64-69, 86, 90, 120-121, 133-137, 145, 183-184, 188, 216-245, 255-257, 273-285, 306-\>308, 310 |
| packages/markitai/src/markitai/fetch\_playwright.py                        |      414 |       75 |      154 |       26 |     78% |104, 109, 121-168, 279-280, 407, 452-456, 461-468, 558, 580, 601-602, 607, 612-\>621, 617-618, 624-625, 631-632, 644-\>686, 672, 675-\>678, 693-699, 700-\>738, 702-\>738, 729-736, 777-780, 823-825, 867, 869, 871, 873, 875, 877, 879, 881, 883, 1041-1042 |
| packages/markitai/src/markitai/fetch\_policy.py                            |       88 |        6 |       52 |        6 |     91% |32, 43, 47, 49, 89, 94 |
| packages/markitai/src/markitai/fetch\_types.py                             |       39 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/image.py                                    |      633 |       80 |      188 |       22 |     86% |109-\>116, 133, 137-139, 191, 248-252, 314-331, 346-381, 406, 465, 479-481, 652, 709, 716-723, 732-\>735, 741, 752-759, 850, 950-952, 1070-1072, 1113-1114, 1131, 1190, 1199, 1272-1274, 1303-1304, 1321, 1410, 1508-\>1536, 1532-1533 |
| packages/markitai/src/markitai/json\_order.py                              |      175 |       24 |      104 |       12 |     82% |226, 231-\>239, 250-\>254, 317-\>321, 384-\>388, 398-\>404, 411-423, 448-\>467, 456-463, 467-\>483, 473-480, 502-\>519 |
| packages/markitai/src/markitai/llm/\_\_init\_\_.py                         |        7 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/cache.py                                |      254 |        8 |       74 |        4 |     96% |181, 407-408, 412-413, 502-\>509, 526-\>exit, 529-530, 540 |
| packages/markitai/src/markitai/llm/content.py                              |      283 |       29 |      138 |       16 |     87% |75, 90-\>86, 292, 301-\>304, 310-339, 350-\>348, 352-\>359, 365-\>364, 368-\>364, 372-\>364, 380-384, 415-\>451, 539-541, 542-\>535, 546-547, 613 |
| packages/markitai/src/markitai/llm/degeneration.py                         |       68 |        0 |       32 |        0 |    100% |           |
| packages/markitai/src/markitai/llm/document.py                             |      681 |       62 |      184 |       26 |     89% |145-147, 151-168, 181, 190-192, 196-\>203, 198-\>196, 200-\>196, 222-223, 365, 453, 485-486, 507, 510, 523, 529, 582, 757-761, 770-\>782, 940-944, 953-\>965, 1212-1216, 1274-\>1305, 1460, 1463-\>1469, 1466, 1473-\>1482, 1646-1677, 1882-1885, 1951, 1956-\>1963, 1967-\>1976, 2026-\>2028 |
| packages/markitai/src/markitai/llm/models.py                               |       66 |        0 |       20 |        2 |     98% |63-\>65, 106-\>110 |
| packages/markitai/src/markitai/llm/processor.py                            |      761 |      146 |      278 |       35 |     79% |44-45, 183-192, 234, 266-\>280, 286-300, 462, 466-475, 516, 539-542, 559-565, 718, 804-805, 835-841, 861, 880-886, 915, 920, 940-950, 990, 994-995, 1020-1026, 1051-1056, 1062, 1079, 1084, 1098-1104, 1267-1270, 1288-\>1286, 1290-\>1292, 1292-\>1286, 1367-1369, 1469-1481, 1515, 1537-1545, 1577-1583, 1615, 1738-\>1744, 1745, 1782-1800, 1817-1818, 1827-1884, 1903-\>1907 |
| packages/markitai/src/markitai/llm/types.py                                |       81 |        5 |       16 |        5 |     90% |156, 159, 169, 171, 176 |
| packages/markitai/src/markitai/llm/vision.py                               |      382 |       20 |      114 |       16 |     92% |91, 99, 157, 463-465, 642, 645-\>653, 657-\>668, 748, 750, 765, 767, 774-778, 882-884, 887-\>894, 898-\>907, 1107-\>1141, 1121-\>1123, 1134-1135, 1141-\>1175, 1155-\>1157, 1168-1169 |
| packages/markitai/src/markitai/ocr.py                                      |      159 |       36 |       36 |        2 |     76% |139-\>149, 193-194, 204-\>223, 302-311, 323-338, 357-381, 405-416, 433-434 |
| packages/markitai/src/markitai/prompts/\_\_init\_\_.py                     |       64 |       12 |       32 |        4 |     77% |100-\>104, 116, 131-\>134, 157-169 |
| packages/markitai/src/markitai/providers/\_\_init\_\_.py                   |      312 |       54 |      126 |       20 |     81% |140, 152-153, 192, 229-231, 278-\>290, 297-\>309, 305-306, 310-\>317, 314-315, 318-320, 399-405, 414-420, 429-\>436, 468-474, 489-495, 507-510, 522-525, 527-\>533, 545, 576, 578, 581, 585, 613-616, 660, 671-673, 712 |
| packages/markitai/src/markitai/providers/auth.py                           |      385 |       65 |      140 |       14 |     82% |50-54, 77-78, 253-254, 275-283, 293-310, 319-323, 344, 436-\>440, 479-480, 555-559, 567-568, 600-601, 642-\>663, 646-651, 652-\>663, 654-\>663, 660-\>663, 664-682, 893, 913, 930-938, 955, 962-964, 1020-1021, 1032 |
| packages/markitai/src/markitai/providers/chatgpt.py                        |      174 |       40 |       64 |       12 |     71% |55-57, 69-71, 88, 117-118, 127, 175-176, 195, 198, 201-202, 209-\>193, 219-223, 232-233, 265-268, 277, 283-\>275, 291, 332-\>341, 347, 359-360, 365-366, 394, 465-471 |
| packages/markitai/src/markitai/providers/claude\_agent.py                  |      178 |       22 |       76 |       16 |     85% |74-78, 100, 162, 167-\>160, 186-190, 222-227, 231-236, 262-\>exit, 305-\>308, 318, 351, 354, 380, 387-\>375, 449-\>448, 451-\>446, 453-\>457, 478-481, 565 |
| packages/markitai/src/markitai/providers/common.py                         |       37 |        2 |       22 |        2 |     93% |27-\>24, 105, 112 |
| packages/markitai/src/markitai/providers/copilot.py                        |      321 |      110 |      102 |       26 |     62% |81-82, 94-98, 114-174, 211, 231-232, 236-273, 304, 306, 311-\>310, 315-\>310, 352-361, 365, 367, 375-\>373, 377-378, 401-405, 417, 461, 465-466, 491, 493-\>533, 502, 567, 570, 579, 604-605, 621-624, 650, 662, 666-670, 673-674, 680, 702-705, 711-712, 731-733, 771-\>781, 805, 810-\>817, 813-814 |
| packages/markitai/src/markitai/providers/errors.py                         |       43 |        1 |        8 |        2 |     94% |284, 305-\>exit |
| packages/markitai/src/markitai/providers/gemini\_cli.py                    |      568 |      121 |      204 |       34 |     74% |75-78, 86-89, 98-100, 105-106, 167, 171, 175, 220, 261, 289-\>300, 293-294, 300-\>310, 313, 317, 319-\>311, 338, 356, 371-374, 382, 408-409, 478-483, 586, 610, 626-\>632, 629-630, 648, 664-\>715, 675, 759, 766-769, 774-783, 795, 809-811, 831-849, 859-882, 892-939, 964, 971, 1020-\>1023, 1036-\>1044, 1074-1077, 1106, 1111-\>1104, 1126-1129, 1170, 1216-\>1223, 1255, 1292, 1321-\>1343, 1331-1338 |
| packages/markitai/src/markitai/providers/json\_mode.py                     |       59 |        7 |       26 |        6 |     85% |109-110, 173-\>188, 175, 179, 181, 183, 185 |
| packages/markitai/src/markitai/providers/oauth\_display.py                 |       69 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/providers/timeout.py                        |       48 |        2 |       26 |        5 |     91% |138, 143-\>135, 147, 152-\>145, 154-\>145 |
| packages/markitai/src/markitai/security.py                                 |      160 |       33 |       50 |        8 |     75% |38-50, 68-80, 125-\>131, 128-129, 137, 196, 270, 289-290, 380, 395, 398-399 |
| packages/markitai/src/markitai/types.py                                    |       15 |       15 |        0 |        0 |      0% |      3-30 |
| packages/markitai/src/markitai/urls.py                                     |       81 |       13 |       38 |        8 |     82% |98-99, 102, 112, 114-115, 128-129, 131-132, 141, 174, 191 |
| packages/markitai/src/markitai/utils/\_\_init\_\_.py                       |       10 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/asset\_store.py                       |       47 |        1 |       14 |        3 |     93% |29, 84-\>90, 87-\>84 |
| packages/markitai/src/markitai/utils/cli\_helpers.py                       |       48 |        1 |       14 |        3 |     94% |53-\>64, 67, 119-\>133 |
| packages/markitai/src/markitai/utils/executor.py                           |       82 |       25 |       28 |        8 |     66% |49-\>54, 75-97, 103-\>102, 107-119, 130, 134, 136, 140 |
| packages/markitai/src/markitai/utils/frontmatter.py                        |      173 |       11 |       84 |       10 |     92% |32, 93, 106, 119-122, 239, 243, 251, 256, 334-\>341, 360-\>352, 415 |
| packages/markitai/src/markitai/utils/guidance.py                           |       28 |        0 |        8 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/markdown\_quality.py                  |        9 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/mime.py                               |       42 |        3 |       18 |        3 |     90% |132, 157, 162 |
| packages/markitai/src/markitai/utils/office.py                             |      119 |       68 |       34 |        5 |     37% |50-56, 78-97, 141-174, 189-\>187, 194-231, 244, 270, 276-278 |
| packages/markitai/src/markitai/utils/output.py                             |       26 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/utils/paths.py                              |       53 |        2 |       10 |        0 |     94% |     24-25 |
| packages/markitai/src/markitai/utils/progress.py                           |       54 |        9 |       20 |        3 |     84% |85-87, 102-104, 109-\>exit, 119, 123-124 |
| packages/markitai/src/markitai/utils/terminal\_image.py                    |       66 |        1 |       22 |        1 |     98% |        83 |
| packages/markitai/src/markitai/utils/text.py                               |      235 |       52 |      110 |       12 |     77% |65, 67-110, 154, 187, 202, 277-\>284, 300, 451, 470-477, 511, 563-564, 587-589, 591-\>553, 596 |
| packages/markitai/src/markitai/webextract/\_\_init\_\_.py                  |       31 |        3 |       10 |        2 |     88% |32, 59, 89 |
| packages/markitai/src/markitai/webextract/constants.py                     |       19 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/dom.py                           |        8 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/\_\_init\_\_.py         |        5 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/callouts.py             |       57 |        1 |       26 |        5 |     93% |34-\>41, 42-\>46, 54-\>61, 74, 77-\>83 |
| packages/markitai/src/markitai/webextract/elements/code.py                 |       44 |        2 |       28 |        2 |     94% |    77, 81 |
| packages/markitai/src/markitai/webextract/elements/footnotes.py            |     1018 |      313 |      564 |       94 |     65% |72-73, 79-80, 123, 136, 146, 148, 155, 162, 182, 186-192, 201-\>197, 209, 213, 277, 291-\>290, 308, 314-315, 328-331, 346, 353, 387-\>386, 414, 417, 421-\>423, 433-442, 448-461, 465-482, 488-504, 510-\>516, 535-551, 573, 596, 598-\>594, 602, 606, 615-\>619, 624, 627-\>632, 655-688, 694-696, 701-744, 779-781, 784, 802-833, 840-\>exit, 859-\>868, 861-864, 880-883, 899, 901-\>895, 903-\>895, 958-\>965, 974, 977, 1002, 1009, 1037, 1048-1049, 1082-\>1079, 1087, 1096-\>1118, 1102-\>1115, 1107-1112, 1114, 1143-1196, 1200, 1209, 1233-\>1240, 1243-1276, 1286-1316, 1330-1359, 1381, 1383, 1392, 1395, 1398, 1403-\>1411, 1421-\>1424, 1425, 1441, 1472, 1474, 1495, 1498, 1501, 1509, 1511, 1514, 1543-\>1542, 1558-\>1563, 1564, 1571-\>1576, 1577, 1584, 1591-1594, 1625-1631 |
| packages/markitai/src/markitai/webextract/elements/headings.py             |       19 |        0 |       10 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/elements/images.py               |       57 |        2 |       26 |        4 |     93% |28-\>31, 72, 90, 91-\>69 |
| packages/markitai/src/markitai/webextract/elements/math.py                 |       69 |        1 |       36 |        5 |     94% |68-\>57, 91-\>93, 97-\>87, 99-\>87, 129 |
| packages/markitai/src/markitai/webextract/enrichers/\_\_init\_\_.py        |        8 |        0 |        0 |        0 |    100% |           |
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
| packages/markitai/src/markitai/webextract/extractors/x\_common.py          |      347 |       46 |      212 |       54 |     81% |90, 94, 96, 127-130, 164, 167-\>161, 173-\>177, 175-\>177, 177-\>182, 206, 219, 239, 243-\>237, 262-\>264, 297, 301-303, 308-\>310, 312, 324, 330, 335, 360, 363, 365, 369, 376, 395, 423, 476, 482, 484-\>479, 486-\>479, 488-\>474, 496, 505-\>494, 508-\>494, 540, 544, 553-554, 559-\>556, 564, 598-\>586, 603, 620, 621-\>618, 641, 646, 647-\>650, 675, 678, 681, 685, 701-\>705, 704, 748-750 |
| packages/markitai/src/markitai/webextract/extractors/x\_tweet.py           |      132 |       14 |       72 |       15 |     85% |63-\>65, 66, 71, 116, 129-\>152, 138, 147-\>136, 150, 180-181, 185-\>188, 204, 219-220, 247, 261-262, 267-\>265, 276 |
| packages/markitai/src/markitai/webextract/extractors/youtube\_page.py      |       97 |       29 |       56 |       17 |     58% |57-58, 83-\>85, 85-\>87, 87-\>90, 122-\>126, 135-151, 172-179, 199-210, 232-\>235, 235-\>238, 240-\>251, 243-\>251 |
| packages/markitai/src/markitai/webextract/frontmatter.py                   |       14 |        0 |        6 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/markdown.py                      |      134 |       13 |       56 |       12 |     87% |89, 94-\>96, 114, 121-124, 177, 180, 218, 226, 234, 266, 272, 288 |
| packages/markitai/src/markitai/webextract/metadata.py                      |      163 |       10 |      102 |       13 |     91% |60, 115, 172, 176-\>179, 181, 205, 238, 241-242, 254, 257-\>256, 261-\>exit, 263-\>262, 279, 282-\>276 |
| packages/markitai/src/markitai/webextract/mobile\_styles.py                |       58 |        5 |       34 |        5 |     89% |28, 38-\>37, 41-42, 68, 72, 81-\>74 |
| packages/markitai/src/markitai/webextract/pipeline.py                      |      191 |       11 |       54 |       11 |     90% |107-\>106, 116, 139-140, 298-\>307, 308-\>315, 315-\>317, 396-398, 400, 425-\>441, 467-469, 471, 503-\>506 |
| packages/markitai/src/markitai/webextract/preprocess.py                    |       54 |        1 |       14 |        1 |     97% |       133 |
| packages/markitai/src/markitai/webextract/quality.py                       |      124 |        3 |       48 |        2 |     97% |87, 135-136 |
| packages/markitai/src/markitai/webextract/removals/\_\_init\_\_.py         |       17 |        0 |        4 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/removals/content\_patterns.py    |       65 |        0 |       32 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/removals/hidden.py               |       61 |        8 |       40 |        8 |     84% |49, 97, 105, 109, 111, 116, 118, 122 |
| packages/markitai/src/markitai/webextract/removals/scoring.py              |      108 |       10 |       66 |        5 |     90% |76, 78-79, 87, 107-108, 128, 163-\>169, 188-190 |
| packages/markitai/src/markitai/webextract/removals/selectors.py            |      120 |       22 |       82 |        8 |     81% |46, 51-64, 84, 87, 101-\>105, 116, 130-132, 159, 164 |
| packages/markitai/src/markitai/webextract/removals/small\_images.py        |       33 |        2 |       16 |        0 |     96% |     57-58 |
| packages/markitai/src/markitai/webextract/render.py                        |      128 |        3 |       64 |        6 |     95% |106, 159, 162, 188-\>194, 213-\>215, 215-\>217 |
| packages/markitai/src/markitai/webextract/resolver.py                      |       55 |        4 |       20 |        1 |     93% |163, 166-172 |
| packages/markitai/src/markitai/webextract/sanitize.py                      |       31 |        0 |       20 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/schema.py                        |       38 |        4 |       20 |        2 |     90% |23, 26-27, 43 |
| packages/markitai/src/markitai/webextract/scoring.py                       |       52 |        0 |       20 |        1 |     99% |   91-\>94 |
| packages/markitai/src/markitai/webextract/semantics.py                     |       38 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/standardize.py                   |      169 |        7 |      104 |        8 |     95% |173, 196, 226, 261, 263, 318, 341, 369-\>367 |
| packages/markitai/src/markitai/webextract/thread\_policy.py                |       16 |        1 |        4 |        1 |     90% |        62 |
| packages/markitai/src/markitai/webextract/types.py                         |       44 |        0 |        0 |        0 |    100% |           |
| packages/markitai/src/markitai/webextract/utils.py                         |       12 |        0 |        2 |        0 |    100% |           |
| packages/markitai/src/markitai/workflow/\_\_init\_\_.py                    |       20 |        4 |        8 |        2 |     79% |37-39, 50-52 |
| packages/markitai/src/markitai/workflow/core.py                            |      483 |       71 |      176 |       37 |     82% |178-\>183, 194-219, 229, 295, 402, 427, 505, 507, 544-\>547, 596, 604-605, 616, 625, 646, 650, 671, 677-\>680, 694-695, 721, 738, 753-\>756, 768, 799, 846-\>855, 909, 949-996, 1023-1024, 1041, 1057, 1064, 1147, 1157, 1162, 1167, 1206-1208, 1213-1214, 1222-1226, 1241-1242, 1254-1255 |
| packages/markitai/src/markitai/workflow/helpers.py                         |      206 |       16 |      106 |       17 |     89% |110, 116, 120, 147-\>149, 150, 278-\>272, 294-\>297, 334-337, 406, 416, 436-437, 439-442, 449-\>448, 452-\>448, 461-\>463, 478-\>484, 482, 534-\>542 |
| packages/markitai/src/markitai/workflow/single.py                          |      196 |        4 |       34 |        6 |     96% |52, 159, 167, 344-\>366, 491 |
| **TOTAL**                                                                  | **21724** | **3263** | **8144** | **1228** | **82%** |           |


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