# This file is used to categorize the necessary icd9 codes for the comorbidities (+ sepsis)

# Complete List of icd9-codes: https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes
# comorbidities (icd9 filtering): https://cumming.ucalgary.ca/sites/default/files/teams/30/resources/Pub_ICD10%E2%80%93ICD9_ElixhauserCharlson_coding_Coding_Algorithm_for_Defining_Comorbidities_in_ICD9-CM_and%20ICD10_Admin_Data.pdf


# sepsis (995.91):
sepsis_list = [99590, 99591, 99592]
# sepsis with acute organ dysfunction (995.92), sepsis with multiple organ dysfunction (995.92), severe sepsis (995.92)


# hypertension: 401.1, 401.9, 402.10, 402.90, 404.10, 404.90, 405.1, 405.9
hypertension_list = [4011, 4019, 40210, 40290, 40410, 40490, 4051, 4059]


# obesity: 278.0
obesity_list = [2780, 27800, 27801, 27802, 2781]
# Non-specific code 278 Overweight, obesity and other hyperalimentation
# Non-specific code 278.0 Overweight and obesity
# Specific code 278.00 Obesity, unspecified convert 278.00 to ICD-10-CM
# Specific code 278.01 Morbid obesity convert 278.01 to ICD-10-CM
# Specific code 278.02 Overweight convert 278.02 to ICD-10-CM
# Specific code 278.03 Obesity hypoventilation syndrome convert 278.03 to ICD-10-CM
# Specific code 278.1 Localized adiposity convert 278.1 to ICD-10-CM


# diabetes: 250.0 - 250.3, 250.4-250.6, 250.7, 250.9
old_diabetes_list = [25000, 25010, 25020, 25030, 25040, 25050, 25060, 25070, 25090]

diabetes_list = [
    24900, 24901, 24910, 24911, 24920, 24921, 24930, 24931, 24940, 24941, 24950, 24951, 24960, 24961, 24970, 24971,
    24980, 24981, 24990, 24991, 25000, 25001, 25002, 25003, 25010, 25011, 25012, 25013, 25020, 25021, 25022, 25023,
    25030, 25031, 25032, 25033, 25040, 25041, 25042, 25043, 25050, 25051, 25052, 25053, 25060, 25061, 25062, 25063,
    25070, 25071, 25072, 25073, 25080, 25081, 25082, 25083, 25090, 25091, 25092, 25093]


# alcohol/drug abuse: 291.x, 303.9, 305.0, V113, 292.0, 292.x, 304.0, 305.2-305.9
drug_abuse_list = [
    2910, 2911, 2912, 2913, 2914, 2915, 29181, 29182, 29189, 2919, 2920, 29211, 29212, 2922, 29281, 29282, 29283,
    29284, 29285, 29289, 2929, 30300, 30301, 30302, 30303, 30390, 30391, 30392, 30393, 30400, 30401, 30402, 30403,
    30410, 30411, 30412, 30413, 30420, 30421, 30422, 30423, 30430, 30431, 30432, 30433, 30440, 30441, 30442, 30443,
    30450, 30451, 30452, 30453, 30460, 30461, 30462, 30463, 30470, 30471, 30472, 30473, 30480, 30481, 30482, 30483,
    30490, 30491, 30492, 30493, 30500, 30501, 30502, 30503, 3051, 30520, 30521, 30522, 30523, 30530, 30531, 30532,
    30533, 30540, 30541, 30542, 30543, 30550, 30551, 30552, 30553, 30560, 30561, 30562, 30563, 30570, 30571, 30572,
    30573, 30580, 30581, 30582, 30583, 30590, 30591, 30592, 30593]


# cancer: 140.x - 172.x, 174.x - 195.8, 200.x - 208.x, 196.x-199.1, V10.x
cancer_list = [
    1400, 1401, 1403, 1404, 1405, 1406, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1418, 1419, 1420, 1421,
    1422, 1428, 1429, 1430, 1431, 1438, 1439, 1440, 1441, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1458,
    1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1478, 1479, 1480, 1481,
    1482, 1483, 1488, 1489, 1490, 1491, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1508, 1509, 1510, 1511, 1512,
    1513, 1514, 1515, 1516, 1518, 1519, 1520, 1521, 1522, 1523, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536,
    1537, 1538, 1539, 1540, 1541, 1542, 1543, 1548, 1550, 1551, 1552, 1560, 1561, 1562, 1568, 1569, 1570, 1571, 1572,
    1573, 1574, 1578, 1579, 1580, 1588, 1589, 1590, 1591, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1608, 1609,
    1610, 1611, 1612, 1613, 1618, 1619, 1620, 1622, 1623, 1624, 1625, 1628, 1629, 1630, 1631, 1638, 1639, 1640, 1641,
    1642, 1643, 1648, 1649, 1650, 1658, 1659, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1712,
    1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 17300, 17301,
    17302, 17309, 17310, 17311, 17312, 17319, 17320, 17321, 17322, 17329, 17330, 17331, 17332, 17339, 17340, 17341,
    17342, 17349, 17350, 17351, 17352, 17359, 17360, 17361, 17362, 17369, 17370, 17371, 17372, 17379, 17380, 17381,
    17382, 17389, 17390, 17391, 17392, 17399, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1748, 1749, 1750, 1759, 1760,
    1761, 1762, 1763, 1764, 1765, 1768, 1769, 179, 1800, 1801, 1808, 1809, 181, 1820, 1821, 1828, 1830, 1832, 1833,
    1834, 1835, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1848, 1849, 185, 1860, 1869, 1871, 1872, 1873, 1874, 1875,
    1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894,
    1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916,
    1917, 1918, 1919, 1920, 1921, 1922, 1923, 1928, 1929, 193, 1940, 1941, 1943, 1944, 1945, 1946, 1948, 1949, 1950,
    1951, 1952, 1953, 1954, 1955, 1958, 1960, 1961, 1962, 1963, 1965, 1966, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
    1975, 1976, 1977, 1978, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 19881, 19882, 19889, 1990, 1992, 1991,
    20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20010, 20011, 20012, 20013, 20014, 20015, 20016,
    20017, 20018, 20020, 20021, 20022, 20023, 20024, 20025, 20026, 20027, 20028, 20030, 20031, 20032, 20033, 20034,
    20035, 20036, 20037, 20038, 20040, 20041, 20042, 20043, 20044, 20045, 20046, 20047, 20048, 20050, 20051, 20052,
    20053, 20054, 20055, 20056, 20057, 20058, 20060, 20061, 20062, 20063, 20064, 20065, 20066, 20067, 20068, 20070,
    20071, 20072, 20073, 20074, 20075, 20076, 20077, 20078, 20080, 20081, 20082, 20083, 20084, 20085, 20086, 20087,
    20088, 20100, 20101, 20102, 20103, 20104, 20105, 20106, 20107, 20108, 20110, 20111, 20112, 20113, 20114, 20115,
    20116, 20117, 20118, 20120, 20121, 20122, 20123, 20124, 20125, 20126, 20127, 20128, 20140, 20141, 20142, 20143,
    20144, 20145, 20146, 20147, 20148, 20150, 20151, 20152, 20153, 20154, 20155, 20156, 20157, 20158, 20160, 20161,
    20162, 20163, 20164, 20165, 20166, 20167, 20168, 20170, 20171, 20172, 20173, 20174, 20175, 20176, 20177, 20178,
    20190, 20191, 20192, 20193, 20194, 20195, 20196, 20197, 20198, 20200, 20201, 20202, 20203, 20204, 20205, 20206,
    20207, 20208, 20210, 20211, 20212, 20213, 20214, 20215, 20216, 20217, 20218, 20220, 20221, 20222, 20223, 20224,
    20225, 20226, 20227, 20228, 20230, 20231, 20232, 20233, 20234, 20235, 20236, 20237, 20238, 20240, 20241, 20242,
    20243, 20244, 20245, 20246, 20247, 20248, 20250, 20251, 20252, 20253, 20254, 20255, 20256, 20257, 20258, 20260,
    20261, 20262, 20263, 20264, 20265, 20266, 20267, 20268, 20270, 20271, 20272, 20273, 20274, 20275, 20276, 20277,
    20278, 20280, 20281, 20282, 20283, 20284, 20285, 20286, 20287, 20288, 20290, 20291, 20292, 20293, 20294, 20295,
    20296, 20297, 20298, 20300, 20301, 20302, 20310, 20311, 20312, 20380, 20381, 20382, 20400, 20401, 20402, 20410,
    20411, 20412, 20420, 20421, 20422, 20480, 20481, 20482, 20490, 20491, 20492, 20500, 20501, 20502, 20510, 20511,
    20512, 20520, 20521, 20522, 20530, 20531, 20532, 20580, 20581, 20582, 20590, 20591, 20592, 20600, 20601, 20602,
    20610, 20611, 20612, 20620, 20621, 20622, 20680, 20681, 20682, 20690, 20691, 20692, 20700, 20701, 20702, 20710,
    20711, 20712, 20720, 20721, 20722, 20780, 20781, 20782, 20800, 20801, 20802, 20810, 20811, 20812, 20820, 20821,
    20822, 20880, 20881, 20882, 20890, 20891, 20892, 20900, 20901, 20902, 20903, 20910, 20911, 20912, 20913, 20914,
    20915, 20916, 20917, 20920, 20921, 20922, 20923, 20924, 20925, 20926, 20927, 20929, 20930, 20931, 20932, 20933,
    20934, 20935, 20936, 20940, 20941, 20942, 20943, 20950, 20951, 20952, 20953, 20954, 20955, 20956, 20957, 20960,
    20961, 20962, 20963, 20964, 20965, 20966, 20967, 20969, 20970, 20971, 20972, 20973, 20974, 20975, 20979,
    'V1000', 'V1001', 'V1002', 'V1003', 'V1004', 'V1005', 'V1006', 'V1007', 'V1009', 'V1011', 'V1012', 'V1020',
    'V1021', 'V1022', 'V1029', 'V103,' 'V1040', 'V1041', 'V1042', 'V1043', 'V1044', 'V1045', 'V1046', 'V1047',
    'V1048', 'V1049', 'V1050', 'V1051', 'V1052', 'V1053', 'V1059', 'V1060', 'V1061', 'V1062', 'V1063', 'V1069',
    'V1071', 'V1072', 'V1079', 'V1081', 'V1082', 'V1083', 'V1084', 'V1085', 'V1086', 'V1087', 'V1088', 'V1089',
    'V1090', 'V1091']