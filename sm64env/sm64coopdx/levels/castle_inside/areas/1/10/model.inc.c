#include "pc/rom_assets.h"
// 0x0702FDD8 - 0x0702FDF0
static const Lights1 inside_castle_seg7_lights_0702FDD8 = gdSPDefLights1(
    0x5f, 0x5f, 0x5f,
    0xff, 0xff, 0xff, 0x28, 0x28, 0x28
);

// 0x0702FDF0 - 0x0702FE70
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_0702FDF0[] = {
    {{{  1422,    614,  -2869}, 0, {  1774,    990}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2038,    614,  -2616}, 0, { -1294,   -274}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1784,    614,  -2869}, 0, {     0,    990}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1169,    614,  -2254}, 0, {  3040,  -2082}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1422,    614,  -2001}, 0, {  1774,  -3346}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2038,    614,  -2254}, 0, { -1294,  -2082}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1784,    614,  -2001}, 0, {     0,  -3346}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1169,    614,  -2616}, 0, {  3040,   -274}, {0x00, 0x7f, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_0702FDF0, 0x00396340, 232834, 0x0002fdf0, 128);
#endif

// 0x0702FE70 - 0x0702FF70
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_0702FE70[] = {
    {{{  1857,    768,  -2073}, 0, {  6834,   2794}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2110,    768,  -2037}, 0, {  9362,   3156}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1965,    768,  -2182}, 0, {  7918,   1710}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2002,    768,  -1928}, 0, {  8278,   4240}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1857,    922,  -2073}, 0, {  6834,   2794}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  2002,    768,  -1928}, 0, {  8278,   4240}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1857,    768,  -2073}, 0, {  6834,   2794}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  2002,    922,  -1928}, 0, {  8278,   4240}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1350,    768,  -2797}, 0, {  1774,  -4430}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1096,    768,  -2833}, 0, {  -752,  -4792}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1241,    768,  -2688}, 0, {   690,  -3346}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1205,    768,  -2942}, 0, {   330,  -5876}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{   553,    614,  -1638}, 0, {     0,   -288}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1422,    614,  -2001}, 0, {  4312,   1498}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1169,    614,  -2254}, 0, {  4312,   -288}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{   807,    614,  -1385}, 0, {     0,   1498}, {0x00, 0x7f, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_0702FE70, 0x00396340, 232834, 0x0002fe70, 240);
#endif

// 0x0702FF70 - 0x0702FFF0
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_0702FF70[] = {
    {{{   590,    614,  -1530}, 0, {  -286,    224}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{   698,    614,  -1421}, 0, {  -286,    990}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{   734,    614,  -1457}, 0, {     0,    990}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{   626,    614,  -1566}, 0, {     0,    224}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1965,    768,  -2688}, 0, {  7918,  -3346}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2001,    768,  -2942}, 0, {  8278,  -5876}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  1857,    768,  -2797}, 0, {  6834,  -4430}, {0x00, 0x7f, 0x00, 0xff}}},
    {{{  2110,    768,  -2833}, 0, {  9362,  -4792}, {0x00, 0x7f, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_0702FF70, 0x00396340, 232834, 0x0002ff60, 112);
#endif

// 0x0702FFF0 - 0x070300E0
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_0702FFF0, 0x00396340, 232834, 0x0002ffd0, 240);

// 0x070300E0 - 0x070301D0
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_070300E0, 0x00396340, 232834, 0x000300c0, 240);

// 0x070301D0 - 0x070302B0
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_070301D0, 0x00396340, 232834, 0x000301b0, 224);

// 0x070302B0 - 0x070303B0
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_070302B0[] = {
    {{{  1965,    768,  -2182}, 0, {  5588,    990}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  2110,    768,  -2037}, 0, {  7632,    990}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  2110,    922,  -2037}, 0, {  7632,   -544}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1784,    614,  -2001}, 0, {  2012,    990}, {0xa7, 0x00, 0xa6, 0xff}}},
    {{{  1784,    922,  -2001}, 0, {  2012,  -2076}, {0xa7, 0x00, 0xa6, 0xff}}},
    {{{  1857,    922,  -2073}, 0, {   990,  -2076}, {0xa7, 0x00, 0xa6, 0xff}}},
    {{{  1857,    768,  -2073}, 0, {   990,   -544}, {0xa7, 0x00, 0xa6, 0xff}}},
    {{{  2038,    922,  -2254}, 0, { -1562,  -2076}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  1881,   1024,  -2194}, 0, {     0,  -3098}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  1833,   1229,  -2339}, 0, {  -716,  -5142}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  2038,    922,  -2254}, 0, { -1562,  -2076}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1965,    922,  -2182}, 0, {  -540,  -2076}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1881,   1024,  -2194}, 0, {     0,  -3098}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1845,   1024,  -2157}, 0, {   478,  -3098}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1857,    922,  -2073}, 0, {   990,  -2076}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1784,    922,  -2001}, 0, {  2012,  -2076}, {0xb6, 0xba, 0xb5, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_070302B0, 0x00396340, 232834, 0x00030290, 256);
#endif

// 0x070303B0 - 0x07030490
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_070303B0[] = {
    {{{  1965,    768,  -2182}, 0, {  5588,    990}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  2110,    922,  -2037}, 0, {  7632,   -544}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1965,    922,  -2182}, 0, {  5588,   -544}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1881,   1024,  -2194}, 0, {     0,  -3098}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  1699,   1229,  -2205}, 0, {  1166,  -5142}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  1833,   1229,  -2339}, 0, {  -716,  -5142}, {0xb6, 0xba, 0xb6, 0xff}}},
    {{{  1965,    922,  -2182}, 0, {  5588,   -544}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  2110,    922,  -2037}, 0, {  7632,   -544}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  2074,   1024,  -2001}, 0, {  7632,  -1566}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1881,   1024,  -2194}, 0, {  4908,  -1566}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1845,   1024,  -2157}, 0, {  4908,  -1566}, {0x50, 0xc8, 0xb0, 0xff}}},
    {{{  2038,   1024,  -1964}, 0, {  7632,  -1566}, {0x50, 0xc8, 0xb0, 0xff}}},
    {{{  2002,    922,  -1928}, 0, {  7632,   -544}, {0x50, 0xc8, 0xb0, 0xff}}},
    {{{  1857,    922,  -2073}, 0, {  5588,   -544}, {0x50, 0xc8, 0xb0, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_070303B0, 0x00396340, 232834, 0x00030390, 224);
#endif

// 0x07030490 - 0x07030590
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030490[] = {
    {{{  1881,   1024,  -2194}, 0, {  7292,    480}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  2074,   1024,  -2001}, 0, { 10018,    480}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  2038,   1024,  -1964}, 0, { 10018,      0}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1845,   1024,  -2157}, 0, {  7292,      0}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1845,   1024,  -2157}, 0, {   478,  -3098}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1784,    922,  -2001}, 0, {  2012,  -2076}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1699,   1229,  -2205}, 0, {  1166,  -5142}, {0xb6, 0xba, 0xb5, 0xff}}},
    {{{  1881,   1024,  -2194}, 0, {     0,  -3098}, {0xb5, 0xba, 0xb7, 0xff}}},
    {{{  1845,   1024,  -2157}, 0, {   478,  -3098}, {0xb5, 0xba, 0xb7, 0xff}}},
    {{{  1699,   1229,  -2205}, 0, {  1166,  -5142}, {0xb5, 0xba, 0xb7, 0xff}}},
    {{{  1507,   1229,  -2665}, 0, {  -716,  -5142}, {0x4b, 0xba, 0x49, 0xff}}},
    {{{  1326,   1024,  -2676}, 0, {   478,  -3098}, {0x4b, 0xba, 0x49, 0xff}}},
    {{{  1362,   1024,  -2713}, 0, {     0,  -3098}, {0x4b, 0xba, 0x49, 0xff}}},
    {{{  1507,   1229,  -2665}, 0, {  -716,  -5142}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1362,   1024,  -2713}, 0, {     0,  -3098}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1422,    922,  -2869}, 0, { -1564,  -2076}, {0x4a, 0xba, 0x4a, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030490, 0x00396340, 232834, 0x00030470, 256);
#endif

// 0x07030590 - 0x07030670
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030590[] = {
    {{{  1507,   1229,  -2665}, 0, {  -716,  -5142}, {0x4b, 0xba, 0x4a, 0xff}}},
    {{{  1374,   1229,  -2531}, 0, {  1166,  -5142}, {0x4b, 0xba, 0x4a, 0xff}}},
    {{{  1326,   1024,  -2676}, 0, {   478,  -3098}, {0x4b, 0xba, 0x4a, 0xff}}},
    {{{  1362,   1024,  -2713}, 0, {     0,  -3098}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1350,    922,  -2797}, 0, {  -542,  -2076}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1422,    922,  -2869}, 0, { -1564,  -2076}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1422,    614,  -2869}, 0, { -1564,    990}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1422,    922,  -2869}, 0, { -1564,  -2076}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1350,    922,  -2797}, 0, {  -542,  -2076}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1350,    768,  -2797}, 0, {  -542,   -542}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1169,    922,  -2616}, 0, {  2012,  -2076}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1241,    922,  -2688}, 0, {   990,  -2076}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1326,   1024,  -2676}, 0, {   478,  -3098}, {0x4a, 0xba, 0x4a, 0xff}}},
    {{{  1374,   1229,  -2531}, 0, {  1166,  -5142}, {0x4a, 0xba, 0x4a, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030590, 0x00396340, 232834, 0x00030570, 240);
#endif

// 0x07030670 - 0x07030760
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030670[] = {
    {{{  1326,   1024,  -2676}, 0, {     0,      0}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1169,   1024,  -2906}, 0, { -2756,    478}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1362,   1024,  -2713}, 0, {     0,    478}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1241,    768,  -2688}, 0, {   990,   -542}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1241,    922,  -2688}, 0, {   990,  -2076}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1169,    922,  -2616}, 0, {  2012,  -2076}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1362,   1024,  -2713}, 0, { -2414,  -1566}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1169,   1024,  -2906}, 0, { -5140,  -1566}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1205,    922,  -2942}, 0, { -5140,   -544}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1350,    922,  -2797}, 0, { -3096,   -544}, {0xb0, 0xc8, 0x50, 0xff}}},
    {{{  1350,    922,  -2797}, 0, { -3096,   -544}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1205,    922,  -2942}, 0, { -5140,   -544}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1205,    768,  -2942}, 0, { -5140,    990}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1350,    768,  -2797}, 0, { -3096,    990}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1133,   1024,  -2869}, 0, { -2756,      0}, {0x00, 0x81, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030670, 0x00396340, 232834, 0x00030660, 224);
#endif

// 0x07030760 - 0x07030860
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030760[] = {
    {{{  1784,    922,  -2001}, 0, { -1052,  -1054}, {0x00, 0x00, 0x81, 0xff}}},
    {{{  1422,    614,  -2001}, 0, {  2560,   2010}, {0x00, 0x00, 0x81, 0xff}}},
    {{{  1422,    922,  -2001}, 0, {  2560,  -1054}, {0x00, 0x00, 0x81, 0xff}}},
    {{{  1241,    768,  -2688}, 0, { -3096,    990}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1096,    922,  -2833}, 0, { -5140,   -542}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1241,    922,  -2688}, 0, { -3096,   -542}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1241,    922,  -2688}, 0, { -3096,   -542}, {0x50, 0xc7, 0xb0, 0xff}}},
    {{{  1096,    922,  -2833}, 0, { -5140,   -542}, {0x50, 0xc7, 0xb0, 0xff}}},
    {{{  1133,   1024,  -2869}, 0, { -5140,  -1564}, {0x50, 0xc7, 0xb0, 0xff}}},
    {{{  1326,   1024,  -2676}, 0, { -2414,  -1566}, {0x50, 0xc7, 0xb0, 0xff}}},
    {{{  1096,    768,  -2833}, 0, { -5140,    990}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{  1241,    768,  -2688}, 0, {   990,   -542}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1169,    922,  -2616}, 0, {  2012,  -2076}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1169,    614,  -2616}, 0, {  2012,    990}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1422,    614,  -2869}, 0, { -1564,    990}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1350,    768,  -2797}, 0, {  -542,   -542}, {0x59, 0x00, 0x59, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030760, 0x00396340, 232834, 0x00030740, 224);
#endif

// 0x07030860 - 0x07030940
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030860[] = {
    {{{  1784,    922,  -2001}, 0, { -1052,  -1054}, {0x00, 0x00, 0x81, 0xff}}},
    {{{  1784,    614,  -2001}, 0, { -1052,   2010}, {0x00, 0x00, 0x81, 0xff}}},
    {{{  1422,    614,  -2001}, 0, {  2560,   2010}, {0x00, 0x00, 0x81, 0xff}}},
    {{{   626,    870,  -1566}, 0, {  1502,   -544}, {0x59, 0x01, 0xa7, 0xff}}},
    {{{   553,    922,  -1638}, 0, {  2524,  -1054}, {0x59, 0x01, 0xa7, 0xff}}},
    {{{   807,    922,  -1385}, 0, { -1052,  -1054}, {0x59, 0x01, 0xa7, 0xff}}},
    {{{  2038,    922,  -2616}, 0, {  2560,  -1054}, {0x81, 0x00, 0x00, 0xff}}},
    {{{  2038,    614,  -2254}, 0, { -1052,   2010}, {0x81, 0x00, 0x00, 0xff}}},
    {{{  2038,    922,  -2254}, 0, { -1052,  -1054}, {0x81, 0x00, 0x00, 0xff}}},
    {{{  2038,    614,  -2616}, 0, {  2560,   2010}, {0x81, 0x00, 0x00, 0xff}}},
    {{{  1169,    922,  -2254}, 0, { -1052,  -1054}, {0x7f, 0x00, 0x00, 0xff}}},
    {{{  1169,    614,  -2254}, 0, { -1052,   2010}, {0x7f, 0x00, 0x00, 0xff}}},
    {{{  1169,    614,  -2616}, 0, {  2560,   2010}, {0x7f, 0x00, 0x00, 0xff}}},
    {{{  1169,    922,  -2616}, 0, {  2560,  -1054}, {0x7f, 0x00, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030860, 0x00396340, 232834, 0x00030820, 240);
#endif

// 0x07030940 - 0x07030A40
#ifdef VERSION_JP
const Vtx inside_castle_seg7_vertex_07030940[] = {
    {{{   807,    922,  -1385}, 0, { -1052,  -1054}, {0x5a, 0xfe, 0xa7, 0xff}}},
    {{{   734,    870,  -1457}, 0, {     0,   -544}, {0x5a, 0xfe, 0xa7, 0xff}}},
    {{{   626,    870,  -1566}, 0, {  1502,   -544}, {0x5a, 0xfe, 0xa7, 0xff}}},
    {{{   626,    870,  -1566}, 0, {  1502,   -544}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{   626,    614,  -1566}, 0, {  1502,   2010}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{   553,    922,  -1638}, 0, {  2524,  -1054}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{  1784,    922,  -2869}, 0, { -1052,  -1054}, {0x00, 0xba, 0x69, 0xff}}},
    {{{  1507,   1229,  -2665}, 0, {  1712,  -4120}, {0x00, 0xba, 0x69, 0xff}}},
    {{{  1422,    922,  -2869}, 0, {  2560,  -1054}, {0x00, 0xba, 0x69, 0xff}}},
    {{{   553,    922,  -1638}, 0, {  2524,  -1054}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{   734,   1126,  -1457}, 0, {     0,  -3098}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{   807,    922,  -1385}, 0, { -1052,  -1054}, {0x59, 0x00, 0xa7, 0xff}}},
    {{{   553,    922,  -1638}, 0, {  2524,  -1054}, {0x5a, 0x00, 0xa7, 0xff}}},
    {{{   626,   1126,  -1566}, 0, {  1500,  -3098}, {0x5a, 0x00, 0xa7, 0xff}}},
    {{{   734,   1126,  -1457}, 0, {     0,  -3098}, {0x5a, 0x00, 0xa7, 0xff}}},
    {{{   553,    614,  -1638}, 0, {  2524,   2010}, {0x59, 0x00, 0xa6, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030940, 0x00396340, 232834, 0x00030910, 224);
#endif

// 0x07030A40 - 0x07030B30
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030A40[] = {
    {{{   807,    922,  -1385}, 0, { -1052,  -1054}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{   807,    614,  -1385}, 0, { -1052,   2010}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{   734,    614,  -1457}, 0, {     0,   2010}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{   734,    870,  -1457}, 0, {     0,   -544}, {0x59, 0x00, 0xa6, 0xff}}},
    {{{  1422,    922,  -2869}, 0, {  2560,  -1054}, {0x00, 0x00, 0x7f, 0xff}}},
    {{{  1784,    614,  -2869}, 0, { -1052,   2010}, {0x00, 0x00, 0x7f, 0xff}}},
    {{{  1784,    922,  -2869}, 0, { -1052,  -1054}, {0x00, 0x00, 0x7f, 0xff}}},
    {{{  1422,    614,  -2869}, 0, {  2560,   2010}, {0x00, 0x00, 0x7f, 0xff}}},
    {{{  1784,    922,  -2869}, 0, { -4118,  -2076}, {0x00, 0xba, 0x69, 0xff}}},
    {{{  1699,   1229,  -2665}, 0, { -2074,  -5142}, {0x00, 0xba, 0x69, 0xff}}},
    {{{  1507,   1229,  -2665}, 0, {  -716,  -5142}, {0x00, 0xba, 0x69, 0xff}}},
    {{{  2038,    922,  -2254}, 0, { -1052,  -1054}, {0x97, 0xba, 0x00, 0xff}}},
    {{{  1833,   1229,  -2339}, 0, {  -206,  -4120}, {0x97, 0xba, 0x00, 0xff}}},
    {{{  2038,    922,  -2616}, 0, {  2560,  -1054}, {0x97, 0xba, 0x00, 0xff}}},
    {{{  1833,   1229,  -2531}, 0, {  1712,  -4120}, {0x97, 0xba, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030A40, 0x00396340, 232834, 0x000309f0, 256);
#endif

// 0x07030B30 - 0x07030C20
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030B30[] = {
    {{{  1422,    922,  -2001}, 0, {  2560,  -1054}, {0x00, 0xba, 0x97, 0xff}}},
    {{{  1507,   1229,  -2205}, 0, {  1712,  -4120}, {0x00, 0xba, 0x97, 0xff}}},
    {{{  1699,   1229,  -2205}, 0, {  -206,  -4120}, {0x00, 0xba, 0x97, 0xff}}},
    {{{  1784,    922,  -2001}, 0, { -1052,  -1054}, {0x00, 0xba, 0x97, 0xff}}},
    {{{   590,    614,  -1530}, 0, {  -542,   2010}, {0x59, 0x00, 0x59, 0xff}}},
    {{{   626,    614,  -1566}, 0, {     0,   2010}, {0x59, 0x00, 0x59, 0xff}}},
    {{{   626,    870,  -1566}, 0, {     0,   -542}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1169,    922,  -2616}, 0, {  2560,  -1054}, {0x69, 0xba, 0x00, 0xff}}},
    {{{  1374,   1229,  -2531}, 0, {  1712,  -4120}, {0x69, 0xba, 0x00, 0xff}}},
    {{{  1169,    922,  -2254}, 0, { -1052,  -1054}, {0x69, 0xba, 0x00, 0xff}}},
    {{{  1374,   1229,  -2339}, 0, {  -206,  -4120}, {0x69, 0xba, 0x00, 0xff}}},
    {{{   698,    870,  -1421}, 0, {     0,      0}, {0x00, 0x81, 0x00, 0xff}}},
    {{{   590,    870,  -1530}, 0, {     0,    990}, {0x00, 0x81, 0x00, 0xff}}},
    {{{   626,    870,  -1566}, 0, {   990,    990}, {0x00, 0x81, 0x00, 0xff}}},
    {{{   734,    870,  -1457}, 0, {   990,      0}, {0x00, 0x81, 0x00, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030B30, 0x00396340, 232834, 0x00030af0, 256);
#endif

// 0x07030C20 - 0x07030D20
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030C20[] = {
    {{{   734,    614,  -1457}, 0, {     0,   2010}, {0xa7, 0x00, 0xa7, 0xff}}},
    {{{   698,    870,  -1421}, 0, {  -542,   -544}, {0xa7, 0x00, 0xa7, 0xff}}},
    {{{   734,    870,  -1457}, 0, {     0,   -544}, {0xa7, 0x00, 0xa7, 0xff}}},
    {{{   698,    614,  -1421}, 0, {  -542,   2010}, {0xa7, 0x00, 0xa7, 0xff}}},
    {{{  2038,    614,  -2616}, 0, { -1052,   2010}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1965,    922,  -2688}, 0, {     0,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1965,    768,  -2688}, 0, {     0,    480}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{   590,    614,  -1530}, 0, {  -542,   2010}, {0x59, 0x00, 0x59, 0xff}}},
    {{{   626,    870,  -1566}, 0, {     0,   -542}, {0x59, 0x00, 0x59, 0xff}}},
    {{{   590,    870,  -1530}, 0, {  -542,   -544}, {0x59, 0x00, 0x59, 0xff}}},
    {{{  1965,    922,  -2688}, 0, {     0,  -1054}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  2038,    922,  -2616}, 0, { -1052,  -1054}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  1881,   1024,  -2676}, 0, {   480,  -2076}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  1881,   1024,  -2676}, 0, {   480,  -2076}, {0xb6, 0xba, 0x4a, 0xff}}},
    {{{  2038,    922,  -2616}, 0, { -1052,  -1054}, {0xb6, 0xba, 0x4a, 0xff}}},
    {{{  1833,   1229,  -2531}, 0, {  -206,  -4120}, {0xb6, 0xba, 0x4a, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030C20, 0x00396340, 232834, 0x00030bf0, 224);
#endif

// 0x07030D20 - 0x07030E20
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030D20[] = {
    {{{  1881,   1024,  -2676}, 0, {   480,  -2076}, {0xb5, 0xba, 0x49, 0xff}}},
    {{{  1699,   1229,  -2665}, 0, {  1676,  -4120}, {0xb5, 0xba, 0x49, 0xff}}},
    {{{  1845,   1024,  -2713}, 0, {   990,  -2076}, {0xb5, 0xba, 0x49, 0xff}}},
    {{{  1881,   1024,  -2676}, 0, {   480,  -2076}, {0xb6, 0xba, 0x4a, 0xff}}},
    {{{  1833,   1229,  -2531}, 0, {  -206,  -4120}, {0xb6, 0xba, 0x4a, 0xff}}},
    {{{  1699,   1229,  -2665}, 0, {  1676,  -4120}, {0xb6, 0xba, 0x4a, 0xff}}},
    {{{  1784,    922,  -2869}, 0, {  2524,  -1054}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  1845,   1024,  -2713}, 0, {   990,  -2076}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  1699,   1229,  -2665}, 0, {  1676,  -4120}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  1857,    922,  -2797}, 0, {  1502,  -1054}, {0xb6, 0xba, 0x4b, 0xff}}},
    {{{  2038,    614,  -2616}, 0, { -1052,   2010}, {0xa6, 0x00, 0x59, 0xff}}},
    {{{  1965,    768,  -2688}, 0, {     0,    480}, {0xa6, 0x00, 0x59, 0xff}}},
    {{{  1857,    768,  -2797}, 0, {  1502,    480}, {0xa6, 0x00, 0x59, 0xff}}},
    {{{  1857,    768,  -2797}, 0, {  1502,    480}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1857,    922,  -2797}, 0, {  1502,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1784,    922,  -2869}, 0, {  2524,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030D20, 0x00396340, 232834, 0x00030cd0, 224);
#endif

// 0x07030E20 - 0x07030F10
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030E20[] = {
    {{{  2038,    614,  -2616}, 0, { -1052,   2010}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  2038,    922,  -2616}, 0, { -1052,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1965,    922,  -2688}, 0, {     0,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1857,    768,  -2797}, 0, {  1502,    480}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1784,    922,  -2869}, 0, {  2524,  -1054}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1784,    614,  -2869}, 0, {  2524,   2010}, {0xa7, 0x00, 0x5a, 0xff}}},
    {{{  1857,    768,  -2797}, 0, {  1502,    480}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1784,    614,  -2869}, 0, {  2524,   2010}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  2038,    614,  -2616}, 0, { -1052,   2010}, {0xa7, 0x00, 0x59, 0xff}}},
    {{{  1857,    768,  -2797}, 0, { -1052,    990}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  2001,    922,  -2942}, 0, {   990,   -544}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  1857,    922,  -2797}, 0, { -1052,   -544}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  1857,    922,  -2797}, 0, { -1052,   -544}, {0x50, 0xc8, 0x50, 0xff}}},
    {{{  2001,    922,  -2942}, 0, {   990,   -544}, {0x50, 0xc8, 0x50, 0xff}}},
    {{{  1845,   1024,  -2713}, 0, { -1734,  -1566}, {0x50, 0xc8, 0x50, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030E20, 0x00396340, 232834, 0x00030db0, 224);
#endif

// 0x07030F10 - 0x07030FF0
#ifdef VERSION_JP
static const Vtx inside_castle_seg7_vertex_07030F10[] = {
    {{{  2001,    922,  -2942}, 0, {   990,   -544}, {0x50, 0xc7, 0x50, 0xff}}},
    {{{  2038,   1024,  -2906}, 0, {   990,  -1566}, {0x50, 0xc7, 0x50, 0xff}}},
    {{{  1845,   1024,  -2713}, 0, { -1734,  -1566}, {0x50, 0xc7, 0x50, 0xff}}},
    {{{  1857,    768,  -2797}, 0, { -1052,    990}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  2001,    768,  -2942}, 0, {   990,    990}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  2001,    922,  -2942}, 0, {   990,   -544}, {0x5a, 0x00, 0x59, 0xff}}},
    {{{  2038,   1024,  -2906}, 0, {  3204,    990}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  2074,   1024,  -2869}, 0, {  3204,    480}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1881,   1024,  -2676}, 0, {   480,    478}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1845,   1024,  -2713}, 0, {   480,    990}, {0x00, 0x81, 0x00, 0xff}}},
    {{{  1881,   1024,  -2676}, 0, { -1734,  -1566}, {0xb0, 0xc8, 0xb0, 0xff}}},
    {{{  2074,   1024,  -2869}, 0, {   990,  -1566}, {0xb0, 0xc8, 0xb0, 0xff}}},
    {{{  1965,    922,  -2688}, 0, { -1052,   -544}, {0xb0, 0xc8, 0xb0, 0xff}}},
    {{{  2110,    922,  -2833}, 0, {   990,   -544}, {0xb0, 0xc8, 0xb0, 0xff}}},
};
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030F10, 0x00396340, 232834, 0x00030e90, 224);
#endif

// 0x07030FF0 - 0x07031070
#ifdef VERSION_JP
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030FF0, 0x00396340, 232834, 0x00030ff0, 128);
#else
ROM_ASSET_LOAD_VTX(inside_castle_seg7_vertex_07030FF0, 0x00396340, 232834, 0x00030f70, 256);
#endif

// 0x07031070 - 0x070310D8
static const Gfx inside_castle_seg7_dl_07031070[] = {
#ifdef VERSION_JP
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09004000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPLight(&inside_castle_seg7_lights_0702FDD8.l, 1),
    gsSPLight(&inside_castle_seg7_lights_0702FDD8.a, 2),
    gsSPVertex(inside_castle_seg7_vertex_0702FDF0, 8, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  4, 0x0),
    gsSP2Triangles( 0,  5,  1, 0x0,  0,  6,  5, 0x0),
    gsSP2Triangles( 0,  7,  3, 0x0,  0,  4,  6, 0x0),
    gsSPEndDisplayList(),
#else
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09004000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPLight(&inside_castle_seg7_lights_0702FDD8.l, 1),
    gsSPLight(&inside_castle_seg7_lights_0702FDD8.a, 2),
    gsSPVertex(inside_castle_seg7_vertex_0702FDF0, 8, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  4, 0x0),
    gsSP2Triangles( 0,  5,  3, 0x0,  0,  2,  6, 0x0),
    gsSP2Triangles( 0,  4,  7, 0x0,  0,  6,  5, 0x0),
    gsSPEndDisplayList(),
#endif
};

// 0x070310D8 - 0x07031168
static const Gfx inside_castle_seg7_dl_070310D8[] = {
#ifdef VERSION_JP
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09005000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPVertex(inside_castle_seg7_vertex_0702FE70, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  1, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  4,  7,  5, 0x0),
    gsSP2Triangles( 8,  9, 10, 0x0,  8, 11,  9, 0x0),
    gsSP2Triangles(12, 13, 14, 0x0, 12, 15, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_0702FF70, 8, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  4,  7,  5, 0x0),
    gsSPEndDisplayList(),
#else
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09005000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPVertex(inside_castle_seg7_vertex_0702FE70, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 3,  6,  4, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles( 7, 10,  8, 0x0, 11, 12, 13, 0x0),
    gsSP1Triangle(11, 14, 12, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_0702FF70, 7, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP1Triangle( 0,  6,  1, 0x0),
    gsSPEndDisplayList(),
#endif
};

// 0x07031168 - 0x07031588
const Gfx inside_castle_seg7_dl_07031168[] = {
#ifdef VERSION_JP
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09003000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPVertex(inside_castle_seg7_vertex_0702FFF0, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles( 3, 10,  4, 0x0, 11, 12, 13, 0x0),
    gsSP1Triangle(11, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070300E0, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP1Triangle(12, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070301D0, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  1,  3,  2, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070302B0, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 3,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070303B0, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030490, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030590, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 10, 12, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030670, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSP1Triangle( 0, 14,  1, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030760, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles( 3, 10,  4, 0x0, 11, 12, 13, 0x0),
    gsSP2Triangles(11, 13, 14, 0x0, 14, 15, 11, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030860, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  9,  7, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030940, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP2Triangles(12, 13, 14, 0x0,  4, 15,  5, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030A40, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  4,  7,  5, 0x0),
    gsSP2Triangles( 8,  9, 10, 0x0, 11, 12, 13, 0x0),
    gsSP1Triangle(12, 14, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030B30, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles( 8, 10,  9, 0x0, 11, 12, 13, 0x0),
    gsSP1Triangle(11, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030C20, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  1, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030D20, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  9,  7, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030E20, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP1Triangle(12, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030F10, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 11, 13, 12, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030FF0, 8, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  1, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  4,  6,  7, 0x0),
    gsSPEndDisplayList(),
#else
    gsDPSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 1, inside_09003000),
    gsDPLoadSync(),
    gsDPLoadBlock(G_TX_LOADTILE, 0, 0, 32 * 32 - 1, CALC_DXT(32, G_IM_SIZ_16b_BYTES)),
    gsSPVertex(inside_castle_seg7_vertex_0702FFF0, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  8,  9, 0x0),
    gsSP2Triangles( 3, 10,  4, 0x0, 11, 12, 13, 0x0),
    gsSP1Triangle(11, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070300E0, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP1Triangle(12, 13, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070301D0, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  1,  3,  2, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070302B0, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 3,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_070303B0, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  9,  7, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030490, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030590, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP2Triangles( 4, 12,  5, 0x0,  9, 13, 14, 0x0),
    gsSP1Triangle( 9, 14, 10, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030670, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  0,  2, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles( 7,  9, 10, 0x0, 11, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030760, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  4,  6,  7, 0x0),
    gsSP2Triangles( 8,  9, 10, 0x0, 11, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030860, 15, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  9,  7, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 10, 12, 13, 0x0),
    gsSP1Triangle( 0,  2, 14, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030940, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  1, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0,  8, 13,  9, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030A40, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  6,  9,  7, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030B30, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 6,  7,  8, 0x0,  9, 10, 11, 0x0),
    gsSP2Triangles(10, 12, 11, 0x0, 13, 14, 15, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030C20, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  3,  4,  5, 0x0),
    gsSP2Triangles( 4,  6,  5, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles( 7, 10,  8, 0x0, 11, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030D20, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0,  7,  9, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030E20, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  3,  1, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles( 7,  9, 10, 0x0, 11, 12, 13, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030F10, 14, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  7,  8,  9, 0x0),
    gsSP2Triangles(10, 11, 12, 0x0,  4, 13,  5, 0x0),
    gsSPVertex(inside_castle_seg7_vertex_07030FF0, 16, 0),
    gsSP2Triangles( 0,  1,  2, 0x0,  0,  2,  3, 0x0),
    gsSP2Triangles( 4,  5,  6, 0x0,  5,  7,  6, 0x0),
    gsSP2Triangles( 8,  9, 10, 0x0,  8, 11,  9, 0x0),
    gsSP2Triangles(12, 13, 14, 0x0, 12, 14, 15, 0x0),
    gsSPEndDisplayList(),
#endif
};

// 0x07031588 - 0x07031608
const Gfx inside_castle_seg7_dl_07031588[] = {
    gsDPPipeSync(),
    gsDPSetCombineMode(G_CC_MODULATERGB, G_CC_MODULATERGB),
    gsSPClearGeometryMode(G_SHADING_SMOOTH),
    gsDPSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, 0, G_TX_LOADTILE, 0, G_TX_WRAP | G_TX_NOMIRROR, G_TX_NOMASK, G_TX_NOLOD, G_TX_WRAP | G_TX_NOMIRROR, G_TX_NOMASK, G_TX_NOLOD),
    gsSPTexture(0xFFFF, 0xFFFF, 0, G_TX_RENDERTILE, G_ON),
    gsDPTileSync(),
    gsDPSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 8, 0, G_TX_RENDERTILE, 0, G_TX_WRAP | G_TX_NOMIRROR, 5, G_TX_NOLOD, G_TX_WRAP | G_TX_NOMIRROR, 5, G_TX_NOLOD),
    gsDPSetTileSize(0, 0, 0, (32 - 1) << G_TEXTURE_IMAGE_FRAC, (32 - 1) << G_TEXTURE_IMAGE_FRAC),
    gsSPDisplayList(inside_castle_seg7_dl_07031070),
    gsSPDisplayList(inside_castle_seg7_dl_070310D8),
    gsSPDisplayList(inside_castle_seg7_dl_07031168),
    gsSPTexture(0xFFFF, 0xFFFF, 0, G_TX_RENDERTILE, G_OFF),
    gsDPPipeSync(),
    gsDPSetCombineMode(G_CC_SHADE, G_CC_SHADE),
    gsSPSetGeometryMode(G_SHADING_SMOOTH),
    gsSPEndDisplayList(),
};
