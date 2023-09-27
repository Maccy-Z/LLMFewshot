from matplotlib import pyplot as plt
import numpy as np

ys = \
[0.0, 0.05827338248491287, 0.06589216738939285, 0.06976331770420074, 0.10204488039016724, 0.1133258044719696, 0.12354917079210281, 0.16470900177955627, 0.1654619723558426, 0.22024005651474, 0.22974614799022675, 0.2609328627586365, 0.26324066519737244, 0.2657490670681, 0.2856155335903168, 0.30347543954849243, 0.3132080137729645, 0.332623153924942, 0.33447933197021484, 0.338392436504364, 0.33934563398361206, 0.3415530323982239, 0.3430580198764801, 0.3488658666610718, 0.35885095596313477, 0.35901114344596863, 0.3649914264678955, 0.36878249049186707, 0.3793548345565796, 0.3860294222831726, 0.38693708181381226, 0.38891270756721497, 0.39216986298561096, 0.4002859890460968, 0.4040771722793579, 0.4071207344532013, 0.41117867827415466, 0.42431408166885376, 0.440279483795166, 0.4408133327960968, 0.44967013597488403, 0.4527346193790436, 0.46499255299568176, 0.46794363856315613, 0.47163233160972595, 0.4728240370750427, 0.4857063293457031, 0.4865575134754181, 0.4987020194530487, 0.500461220741272, 0.5084629654884338, 0.5135137438774109, 0.5182239413261414, 0.5192453861236572, 0.5217423439025879, 0.5224234461784363, 0.527076780796051, 0.5313331484794617, 0.5402995944023132, 0.5441018342971802, 0.5444991588592529, 0.5513657927513123, 0.5521035194396973, 0.5629614591598511, 0.5711527466773987, 0.5800065994262695, 0.5801872611045837, 0.5808498859405518, 0.5880773067474365, 0.5956060290336609, 0.5977141261100769, 0.5979551076889038, 0.6006051898002625, 0.6014485359191895, 0.6037371158599854, 0.6042190790176392, 0.6046407222747803, 0.6054235696792603, 0.6066281795501709, 0.6074715256690979, 0.6079533100128174, 0.6152411699295044, 0.6173490881919861, 0.6179514527320862, 0.6237937211990356, 0.6268051862716675, 0.6304190158843994, 0.6480664610862732, 0.6487289667129517, 0.6496925950050354, 0.650776743888855, 0.6525234580039978, 0.654812216758728, 0.6559565663337708, 0.6591487526893616, 0.6625818610191345, 0.6709371209144592, 0.6726589202880859, 0.6812680959701538, 0.6821609735488892, 0.6829900145530701, 0.6854133605957031, 0.6882829666137695, 0.6894946098327637, 0.6900685429573059, 0.6911527514457703, 0.6921731233596802, 0.6945964097976685, 0.7028229832649231, 0.709455132484436, 0.7160874009132385, 0.7194035649299622, 0.7217630743980408, 0.7368132472038269, 0.7379611134529114, 0.7435092926025391, 0.7503966093063354, 0.7510342597961426, 0.7526923418045044, 0.7528836727142334, 0.7530750036239624, 0.7532025575637817, 0.7559446692466736, 0.7582404613494873, 0.7590057849884033, 0.7595797181129456, 0.7620667815208435, 0.7644901275634766, 0.768699049949646, 0.7704208493232727, 0.7790938019752502, 0.7798590660095215, 0.7825374603271484, 0.7870652079582214, 0.789106011390686, 0.7960564494132996, 0.796999454498291, 0.7992894053459167, 0.8084496259689331, 0.8145114779472351, 0.8171382546424866, 0.8191589117050171, 0.8325623273849487, 0.8381527662277222, 0.8451575636863708, 0.8459658622741699, 0.8467741012573242, 0.8505458831787109, 0.8518257141113281, 0.8532400727272034, 0.8549912571907043, 0.8566077947616577, 0.8571466207504272, 0.859234631061554, 0.8599754571914673, 0.8618614077568054, 0.8630737066268921, 0.8634105324745178, 0.8662393689155579, 0.8671823740005493, 0.8678559064865112, 0.8724360466003418, 0.8729748129844666, 0.8735809922218323, 0.874254584312439, 0.8789693713188171, 0.8802490234375, 0.8817981481552124, 0.884290337562561, 0.8865803480148315, 0.8869845271110535, 0.8942587375640869, 0.8949995636940002, 0.8979631662368774, 0.8997817039489746, 0.9036209583282471, 0.9061130285263062, 0.9066519141197205, 0.9077968597412109, 0.9090766310691833, 0.9127137660980225, 0.915003776550293, 0.920101523399353, 0.9225142002105713, 0.9309591054916382, 0.9447261691093445, 0.9457196593284607, 0.9521064162254333, 0.9566481709480286, 0.9579965472221375, 0.9590609669685364, 0.9610479474067688, 0.9641703367233276, 0.9686411619186401, 0.9714797735214233, 0.9781504273414612, 0.9816276431083679, 0.9852467775344849, 0.9880853295326233, 0.9917045831680298, 0.9934786558151245, 0.9951108694076538, 0.9964591264724731, 0.9982332587242126, 0.9983043074607849, 1.0013556480407715, 1.0051168203353882, 1.01121985912323, 1.0157614946365356, 1.015903353691101, 1.0171098709106445, 1.0178195238113403, 1.0220773220062256, 1.0259803533554077, 1.0289608240127563, 1.0473406314849854, 1.049256682395935, 1.0523791313171387, 1.0556544065475464, 1.0585635900497437, 1.0596826076507568, 1.0663961172103882, 1.0677390098571777, 1.0683356523513794, 1.0737812519073486, 1.0825090408325195, 1.0863133668899536, 1.0909383296966553, 1.0937728881835938, 1.0973535776138306, 1.098696231842041, 1.1107808351516724, 1.1126457452774048, 1.1177928447723389, 1.1200307607650757, 1.1285346746444702, 1.1312947273254395, 1.132264494895935, 1.1323391199111938, 1.1353975534439087, 1.1372624635696411, 1.1379337310791016, 1.1400225162506104, 1.1422603130340576, 1.1441997289657593, 1.146437644958496, 1.1488993167877197, 1.1522561311721802, 1.1547179222106934, 1.1575524806976318, 1.1576271057128906, 1.159641146659851, 1.1634455919265747, 1.163520097732544, 1.1658326387405396, 1.1682941913604736, 1.1700845956802368, 1.1713526248931885, 1.173366904258728, 1.175530195236206, 1.175828456878662, 1.1759777069091797, 1.1817216873168945, 1.1829897165298462, 1.1921648979187012, 1.1955219507217407, 1.2006796598434448, 1.2013055086135864, 1.2024791240692139, 1.2107725143432617, 1.2117897272109985, 1.214371681213379, 1.2150758504867554, 1.2234476804733276, 1.231350064277649, 1.231741189956665, 1.2333842515945435, 1.2335407733917236, 1.2358096837997437, 1.2359662055969238, 1.2383134365081787, 1.2406606674194336, 1.240895390510559, 1.241834282875061, 1.2458245754241943, 1.2488759756088257, 1.2490324974060059, 1.2531791925430298, 1.2587344646453857, 1.2631940841674805, 1.2649153470993042, 1.2661672830581665, 1.2663238048553467, 1.2666367292404175, 1.2671061754226685, 1.2721918821334839, 1.2723482847213745, 1.2725048065185547, 1.2737566232681274, 1.2740696668624878, 1.278059959411621, 1.28244149684906, 1.2915173768997192, 1.3009843826293945, 1.3055224418640137, 1.3073220252990723, 1.3084956407546997, 1.3135813474655151, 1.3157720565795898, 1.3178062438964844, 1.322109580039978, 1.3268040418624878, 1.333611011505127, 1.334080457687378, 1.3352540731430054, 1.3354886770248413, 1.3368189334869385, 1.3372100591659546, 1.338461995124817, 1.3422174453735352, 1.3476097583770752, 1.350639820098877, 1.3523597717285156, 1.3582563400268555, 1.364644169807434, 1.3677563667297363, 1.3728338479995728, 1.3775838613510132, 1.3796312808990479, 1.3812693357467651, 1.3933080434799194, 1.3972389698028564, 1.400105357170105, 1.4026442766189575, 1.408868432044983, 1.41279935836792, 1.4137821197509766, 1.4160752296447754, 1.4177131652832031, 1.424674391746521, 1.4302434921264648, 1.431226134300232, 1.4317175149917603, 1.4319632053375244, 1.442445993423462, 1.449570894241333, 1.4513726234436035, 1.4547303915023804, 1.4649674892425537, 1.4709458351135254, 1.4717649221420288, 1.4810192584991455, 1.4829027652740479, 1.4836398363113403, 1.4850322008132935, 1.486342430114746, 1.5022542476654053, 1.5069602727890015, 1.5092705488204956, 1.5108107328414917, 1.514490008354187, 1.5192816257476807, 1.5209929943084717, 1.5222764015197754, 1.524672269821167, 1.5270681381225586, 1.528180480003357, 1.528950572013855, 1.5291216373443604, 1.5300629138946533, 1.5311752557754517, 1.5325442552566528, 1.533228874206543, 1.536651372909546, 1.5400739908218384, 1.5438388586044312, 1.548972725868225, 1.5530799627304077, 1.5541067123413086, 1.5616363286972046, 1.5635188817977905, 1.5705351829528809, 1.570962905883789, 1.5739578008651733, 1.5757546424865723, 1.5778082609176636, 1.5786638259887695, 1.5810596942901611, 1.5846534967422485, 1.58987295627594, 1.5908998250961304, 1.5935521125793457, 1.6011674404144287, 1.602450966835022, 1.6053601503372192, 1.6064724922180176, 1.6070715188980103, 1.6077560186386108, 1.6109219789505005, 1.6131465435028076, 1.6143444776535034, 1.615799069404602, 1.6177672147750854, 1.6238421201705933, 1.6242700815200806, 1.6258102655410767, 1.628462791442871, 1.630345106124878, 1.6321420669555664, 1.6347945928573608, 1.6365058422088623, 1.6425811052322388, 1.6458324193954468, 1.6470303535461426, 1.6529343128204346, 1.659095048904419, 1.6595230102539062, 1.6608062982559204, 1.6652039289474487, 1.6776070594787598, 1.6779640913009644, 1.678856372833252, 1.6891179084777832, 1.6915271282196045, 1.6946501731872559, 1.6975055932998657, 1.7034839391708374, 1.7064286470413208, 1.7066072225570679, 1.7080347537994385, 1.7099978923797607, 1.711425542831421, 1.7265948057174683, 1.728111743927002, 1.7299855947494507, 1.731680989265442, 1.7371240854263306, 1.740247130393982, 1.748634696006775, 1.7504194974899292, 1.7509548664093018, 1.7532747983932495, 1.756844162940979, 1.7626441717147827, 1.7637147903442383, 1.7799547910690308, 1.7811146974563599, 1.784773349761963, 1.7857547998428345, 1.7870932817459106, 1.789502501487732, 1.7907516956329346, 1.7927147150039673, 1.7951239347457886, 1.7976224422454834, 1.798693299293518, 1.7990500926971436, 1.8124347925186157, 1.8193947076797485, 1.822964072227478, 1.8252840042114258, 1.832263469696045, 1.8355144262313843, 1.839229702949524, 1.8419233560562134, 1.8440595865249634, 1.8458243608474731, 1.846288800239563, 1.848425030708313, 1.857434630393982, 1.8586421012878418, 1.8661656379699707, 1.8723888397216797, 1.8806551694869995, 1.885485053062439, 1.8878999948501587, 1.888085961341858, 1.8979313373565674, 1.9097275733947754, 1.9275609254837036, 1.9290469884872437, 1.930068850517273, 1.938706874847412, 1.9453942775726318, 1.9454872608184814, 1.9463231563568115, 1.950317144393921, 1.9572833776474, 1.9585837125778198, 1.96239173412323, 1.9626704454421997, 1.9633206129074097, 1.9677790403366089, 1.969450831413269, 1.978553295135498, 1.9792964458465576, 1.9815256595611572, 1.9817113876342773, 1.9822688102722168, 1.9833834171295166, 1.9840335845947266, 1.9857983589172363, 1.9881203174591064, 1.9884917736053467, 1.9967583417892456, 2.000659465789795, 2.0012168884277344, 2.0037245750427246, 2.0204403400421143, 2.022852897644043, 2.0265204906463623, 2.0325043201446533, 2.033179759979248, 2.034048318862915, 2.0393567085266113, 2.041576385498047, 2.0547983646392822, 2.0566322803497314, 2.07004714012146, 2.0732321739196777, 2.07603120803833, 2.082979917526245, 2.083751916885376, 2.0901217460632324, 2.092148542404175, 2.0996763706207275, 2.100738048553467, 2.101896047592163, 2.1030545234680176, 2.104405403137207, 2.1048879623413086, 2.1057565212249756, 2.111933469772339, 2.1140565872192383, 2.1169519424438477, 2.122453212738037, 2.1244800090789795, 2.1255414485931396, 2.1287264823913574, 2.137026309967041, 2.1391496658325195, 2.1401147842407227, 2.143782377243042, 2.1518893241882324, 2.1543021202087402, 2.159996271133423, 2.165208101272583, 2.177271842956543, 2.182290554046631, 2.183255672454834, 2.2017688751220703, 2.206573724746704, 2.2080750465393066, 2.2100770473480225, 2.2108778953552246, 2.2200872898101807, 2.2236907482147217, 2.2376046180725098, 2.2393064498901367, 2.2435104846954346, 2.2471141815185547, 2.250317335128784, 2.2563233375549316, 2.2591261863708496, 2.260127305984497, 2.2706377506256104, 2.2778449058532715, 2.2826497554779053, 2.293660879135132, 2.295562744140625, 2.2968640327453613, 2.3018691539764404, 2.3168840408325195, 2.3174846172332764, 2.3268940448760986, 2.330097198486328, 2.333200454711914, 2.3345017433166504, 2.3425095081329346, 2.349316358566284, 2.351919174194336, 2.3621292114257812, 2.367734909057617, 2.3853812217712402, 2.3997836112976074, 2.4043426513671875, 2.416776657104492, 2.4223718643188477, 2.4285888671875, 2.4313862323760986, 2.4335622787475586, 2.4349093437194824, 2.441230058670044, 2.4415407180786133, 2.4422659873962402, 2.4441311359405518, 2.445167303085327, 2.4521095752716064, 2.4544928073883057, 2.466719150543213, 2.4717965126037598, 2.47635555267334, 2.4777026176452637, 2.482572555541992, 2.4828834533691406, 2.4866135120391846, 2.4933483600616455, 2.5028810501098633, 2.507750988006592, 2.5112738609313965, 2.5149004459381104, 2.5187342166900635, 2.519770383834839, 2.5293030738830566, 2.5322043895721436, 2.5345873832702637, 2.5352091789245605, 2.545156240463257, 2.5507514476776123, 2.55810809135437, 2.5627708435058594, 2.566190242767334, 2.5710601806640625, 2.571474552154541, 2.578002452850342, 2.5806658267974854, 2.5855894088745117, 2.5986475944519043, 2.6109566688537598, 2.620375871658325, 2.620911121368408, 2.6211252212524414, 2.6243362426757812, 2.628831624984741, 2.6440305709838867, 2.6479909420013428, 2.648419141769409, 2.6518442630767822, 2.6711106300354004, 2.6745357513427734, 2.6787099838256836, 2.6912331581115723, 2.695300579071045, 2.701936721801758, 2.7051477432250977, 2.706111192703247, 2.706967353820801, 2.7096431255340576, 2.717992067337036, 2.7220592498779297, 2.7247352600097656, 2.7261266708374023, 2.733083963394165, 2.7358670234680176, 2.7427170276641846, 2.7447509765625, 2.7486040592193604, 2.7869749069213867, 2.789843797683716, 2.8066163063049316, 2.8206303119659424, 2.8263683319091797, 2.8409337997436523, 2.8417062759399414, 2.8425891399383545, 2.8631131649017334, 2.8639960289001465, 2.8671960830688477, 2.8674166202545166, 2.872382164001465, 2.877237558364868, 2.8796651363372803, 2.8838582038879395, 2.890368700027466, 2.893347978591919, 2.895334243774414, 2.912327527999878, 2.912548065185547, 2.916520357131958, 2.918286085128784, 2.9262309074401855, 2.9266722202301025, 2.936713695526123, 2.9477481842041016, 2.952824115753174, 2.975334405899048, 2.977762222290039, 2.9939489364624023, 2.9982635974884033, 3.004054546356201, 3.0068931579589844, 3.010867118835449, 3.0155227184295654, 3.0187020301818848, 3.0194966793060303, 3.0229032039642334, 3.03232741355896, 3.03448486328125, 3.0383455753326416, 3.0423195362091064, 3.044590473175049, 3.0459530353546143, 3.0494730472564697, 3.056058645248413, 3.058102607727051, 3.067754030227661, 3.0802438259124756, 3.093642473220825, 3.0965945720672607, 3.101022958755493, 3.1054511070251465, 3.1300907135009766, 3.1401963233947754, 3.1409912109375, 3.1459872722625732, 3.162224292755127, 3.1685829162597656, 3.1790292263031006, 3.1792562007904053, 3.180164337158203, 3.2000348567962646, 3.201624870300293, 3.206847667694092, 3.212095260620117, 3.228074789047241, 3.2316906452178955, 3.2447540760040283, 3.269364356994629, 3.272747039794922, 3.2940919399261475, 3.3051724433898926, 3.3122873306274414, 3.321385145187378, 3.32698392868042, 3.3359646797180176, 3.3398139476776123, 3.3505446910858154, 3.403731346130371, 3.4124791622161865, 3.42344331741333, 3.4467649459838867, 3.449516773223877, 3.4575328826904297, 3.460643768310547, 3.48134183883667, 3.4937844276428223, 3.502518653869629, 3.505629301071167, 3.51938796043396, 3.5242931842803955, 3.5314717292785645, 3.542957305908203, 3.5526485443115234, 3.575859308242798, 3.582319974899292, 3.585669994354248, 3.586028814315796, 3.5915322303771973, 3.606607437133789, 3.6153411865234375, 3.617375373840332, 3.6233572959899902, 3.6331677436828613, 3.634125232696533, 3.638073205947876, 3.641901969909668, 3.6432178020477295, 3.6458499431610107, 3.6471662521362305, 3.652071237564087, 3.6533875465393066, 3.65627384185791, 3.6570093631744385, 3.658970832824707, 3.665344715118408, 3.6834864616394043, 3.7158472537994385, 3.730189085006714, 3.7446534633636475, 3.7486984729766846, 3.7532339096069336, 3.7593626976013184, 3.7726011276245117, 3.8313164710998535, 3.852522611618042, 3.863677501678467, 3.8675999641418457, 3.8748319149017334, 3.9204304218292236, 3.9438910484313965, 3.961455821990967, 3.971116304397583, 3.98980975151062, 3.9995954036712646, 4.002355575561523, 4.002480983734131, 4.005868434906006, 4.0086283683776855, 4.011137962341309, 4.05454683303833, 4.063454627990723, 4.082273483276367, 4.096074104309082, 4.097579479217529, 4.107992649078369, 4.139002799987793, 4.174286365509033, 4.175954341888428, 4.23459005355835, 4.236514568328857, 4.240748405456543, 4.241133213043213, 4.243314743041992, 4.257043361663818, 4.268077373504639, 4.282062530517578, 4.292070388793945, 4.300025463104248, 4.302719593048096, 4.318500995635986, 4.320939064025879, 4.322093486785889, 4.3231201171875, 4.3274827003479, 4.333127975463867, 4.334539413452148, 4.356736183166504, 4.358788967132568, 4.372039318084717, 4.378857135772705, 4.409144878387451, 4.414127349853516, 4.450577259063721, 4.491091251373291, 4.517970085144043, 4.529114723205566, 4.540128707885742, 4.571202754974365, 4.58287239074707, 4.591001033782959, 4.619988918304443, 4.6206583976745605, 4.627085208892822, 4.630967617034912, 4.634583473205566, 4.647303104400635, 4.647972583770752, 4.652792453765869, 4.704608917236328, 4.707822322845459, 4.715186595916748, 4.718265533447266, 4.760977268218994, 4.765395641326904, 4.804224967956543, 4.823371410369873, 4.832744121551514, 4.836760520935059, 4.841312885284424, 4.868626594543457, 4.902683734893799, 4.9050068855285645, 4.909516334533691, 4.9162116050720215, 4.985081672668457, 5.036870956420898, 5.079368591308594, 5.093716144561768, 5.104921340942383, 5.124735355377197, 5.152441024780273, 5.167494773864746, 5.185474872589111, 5.261299133300781, 5.267850399017334, 5.2851338386535645, 5.32304573059082, 5.345625877380371, 5.397336959838867, 5.41167688369751, 5.431002616882324, 5.435976028442383, 5.4780378341674805, 5.501200199127197, 5.529478549957275, 5.534167766571045, 5.551788330078125, 5.557045936584473, 5.627669811248779, 5.660778999328613, 5.675414562225342, 5.701623439788818, 5.801392078399658, 5.807329177856445, 5.828759670257568, 5.894789695739746, 5.954219818115234, 5.987256050109863, 6.006871700286865, 6.128840923309326, 6.152143478393555, 6.193881511688232, 6.219248294830322, 6.2651286125183105, 6.285698890686035, 6.315279960632324, 6.359274864196777, 6.393360614776611, 6.423090934753418, 6.4402079582214355, 6.507778167724609, 6.593502044677734, 6.597628116607666, 6.634457111358643, 6.70949125289917, 6.84445333480835, 6.99542236328125, 7.026517868041992, 7.100325584411621, 7.148713111877441, 7.206272602081299, 7.206905364990234, 7.27442741394043, 7.288500785827637, 7.291979789733887, 7.307793140411377, 7.328350067138672, 7.341158390045166, 7.570187091827393, 7.573402404785156, 7.575492858886719, 7.593499660491943, 7.948835372924805, 7.956026077270508, 7.971060752868652, 8.132511138916016, 8.321820259094238, 8.393970489501953, 8.403755187988281, 8.511046409606934, 8.547316551208496, 8.679250717163086, 8.800040245056152, 8.853154182434082, 9.000954627990723, 9.049144744873047, 9.13560676574707, 9.17109489440918, 9.250773429870605, 9.62793254852295, 9.763121604919434, 10.273743629455566, 10.329861640930176, 10.882973670959473, 11.520487785339355, 11.762421607971191, 12.32673168182373, 12.454997062683105, 12.484246253967285, 13.050376892089844, 13.181644439697266, 13.202136993408203, 13.89484977722168, 15.436017036437988, 17.381898880004883, 22.585041046142578]
xs = \
[0.536, 0.6775, 0.696, 0.7054, 0.78, 0.8056, 0.8288, 0.9218, 0.9234, 1.0398, 1.06, 1.125, 1.1296, 1.1346, 1.1742, 1.2098, 1.2292, 1.2679, 1.2716, 1.2794, 1.2813, 1.2857, 1.2887, 1.3, 1.3187, 1.319, 1.3302, 1.3373, 1.3571, 1.3696, 1.3713, 1.375, 1.3811, 1.3963, 1.4034, 1.4091, 1.4167, 1.4413, 1.4712, 1.4722, 1.4886, 1.494, 1.5156, 1.5208, 1.5273, 1.5294, 1.5521, 1.5536, 1.575, 1.5781, 1.5922, 1.6011, 1.6094, 1.6112, 1.6156, 1.6168, 1.625, 1.6325, 1.6483, 1.655, 1.6557, 1.6678, 1.6691, 1.6875, 1.7011, 1.7158, 1.7161, 1.7172, 1.7292, 1.7417, 1.7452, 1.7456, 1.75, 1.7514, 1.7552, 1.756, 1.7567, 1.758, 1.76, 1.7614, 1.7622, 1.7743, 1.7778, 1.7788, 1.7885, 1.7935, 1.7995, 1.8288, 1.8299, 1.8315, 1.8333, 1.8362, 1.84, 1.8419, 1.8472, 1.8529, 1.8667, 1.8694, 1.8829, 1.8843, 1.8856, 1.8894, 1.8939, 1.8958, 1.8967, 1.8984, 1.9, 1.9038, 1.9167, 1.9271, 1.9375, 1.9427, 1.9464, 1.97, 1.9718, 1.9805, 1.9913, 1.9923, 1.9949, 1.9952, 1.9955, 1.9957, 2.0, 2.0036, 2.0048, 2.0057, 2.0096, 2.0134, 2.02, 2.0227, 2.0363, 2.0375, 2.0417, 2.0488, 2.052, 2.0625, 2.0639, 2.0673, 2.0809, 2.0899, 2.0938, 2.0968, 2.1167, 2.125, 2.1354, 2.1366, 2.1378, 2.1434, 2.1453, 2.1474, 2.15, 2.1524, 2.1532, 2.1563, 2.1574, 2.1602, 2.162, 2.1625, 2.1667, 2.1681, 2.1691, 2.1759, 2.1767, 2.1776, 2.1786, 2.1856, 2.1875, 2.1898, 2.1935, 2.1969, 2.1975, 2.2083, 2.2094, 2.2138, 2.2165, 2.2222, 2.2259, 2.2267, 2.2284, 2.2303, 2.2357, 2.2391, 2.2466, 2.25, 2.2619, 2.2813, 2.2827, 2.2917, 2.2981, 2.3, 2.3015, 2.3043, 2.3087, 2.315, 2.319, 2.3284, 2.3333, 2.3384, 2.3424, 2.3475, 2.35, 2.3523, 2.3542, 2.3567, 2.3568, 2.3611, 2.3664, 2.375, 2.3814, 2.3816, 2.3833, 2.3843, 2.3903, 2.3958, 2.4, 2.4259, 2.4286, 2.433, 2.4375, 2.4414, 2.4429, 2.4519, 2.4537, 2.4545, 2.4618, 2.4735, 2.4786, 2.4848, 2.4886, 2.4934, 2.4952, 2.5114, 2.5139, 2.5208, 2.5238, 2.5352, 2.5389, 2.5402, 2.5403, 2.5444, 2.5469, 2.5478, 2.5506, 2.5536, 2.5562, 2.5592, 2.5625, 2.567, 2.5703, 2.5741, 2.5742, 2.5769, 2.582, 2.5821, 2.5852, 2.5885, 2.5909, 2.5926, 2.5953, 2.5982, 2.5986, 2.5988, 2.6065, 2.6082, 2.6205, 2.625, 2.6316, 2.6324, 2.6339, 2.6445, 2.6458, 2.6491, 2.65, 2.6607, 2.6708, 2.6713, 2.6734, 2.6736, 2.6765, 2.6767, 2.6797, 2.6827, 2.683, 2.6842, 2.6893, 2.6932, 2.6934, 2.6987, 2.7058, 2.7115, 2.7137, 2.7153, 2.7155, 2.7159, 2.7165, 2.723, 2.7232, 2.7234, 2.725, 2.7254, 2.7305, 2.7361, 2.7477, 2.7598, 2.7656, 2.7679, 2.7694, 2.7759, 2.7787, 2.7813, 2.7868, 2.7928, 2.8015, 2.8021, 2.8036, 2.8039, 2.8056, 2.8061, 2.8077, 2.8125, 2.8192, 2.8229, 2.825, 2.8322, 2.84, 2.8438, 2.85, 2.8558, 2.8583, 2.8603, 2.875, 2.8798, 2.8833, 2.8864, 2.894, 2.8988, 2.9, 2.9028, 2.9048, 2.9133, 2.9201, 2.9213, 2.9219, 2.9222, 2.935, 2.9437, 2.9459, 2.95, 2.9625, 2.9698, 2.9708, 2.9821, 2.9844, 2.9853, 2.987, 2.9886, 3.0079, 3.0134, 3.0161, 3.0179, 3.0222, 3.0278, 3.0298, 3.0313, 3.0341, 3.0369, 3.0382, 3.0391, 3.0393, 3.0404, 3.0417, 3.0433, 3.0441, 3.0481, 3.0521, 3.0565, 3.0625, 3.0673, 3.0685, 3.0773, 3.0795, 3.0877, 3.0882, 3.0917, 3.0938, 3.0962, 3.0972, 3.1, 3.1042, 3.1103, 3.1115, 3.1146, 3.1235, 3.125, 3.1284, 3.1297, 3.1304, 3.1312, 3.1349, 3.1375, 3.1389, 3.1406, 3.1429, 3.15, 3.1505, 3.1523, 3.1554, 3.1576, 3.1597, 3.1628, 3.1648, 3.1719, 3.1757, 3.1771, 3.184, 3.1912, 3.1917, 3.1932, 3.1982, 3.2121, 3.2125, 3.2135, 3.225, 3.2277, 3.2312, 3.2344, 3.2411, 3.2444, 3.2446, 3.2462, 3.2484, 3.25, 3.267, 3.2687, 3.2708, 3.2727, 3.2788, 3.2823, 3.2917, 3.2937, 3.2943, 3.2969, 3.3009, 3.3074, 3.3086, 3.3268, 3.3281, 3.3322, 3.3333, 3.3348, 3.3375, 3.3389, 3.3411, 3.3438, 3.3466, 3.3478, 3.3482, 3.3632, 3.371, 3.375, 3.3776, 3.3854, 3.3889, 3.3929, 3.3958, 3.3981, 3.4, 3.4005, 3.4028, 3.4125, 3.4138, 3.4219, 3.4286, 3.4375, 3.4427, 3.4453, 3.4455, 3.4561, 3.4688, 3.488, 3.4896, 3.4907, 3.5, 3.5072, 3.5073, 3.5082, 3.5125, 3.52, 3.5214, 3.5255, 3.5258, 3.5265, 3.5313, 3.5331, 3.5429, 3.5437, 3.5461, 3.5463, 3.5469, 3.5481, 3.5488, 3.5507, 3.5532, 3.5536, 3.5625, 3.5667, 3.5673, 3.57, 3.5875, 3.59, 3.5938, 3.6, 3.6007, 3.6016, 3.6071, 3.6094, 3.6231, 3.625, 3.6389, 3.6422, 3.6451, 3.6523, 3.6531, 3.6597, 3.6618, 3.6696, 3.6707, 3.6719, 3.6731, 3.6745, 3.675, 3.6759, 3.6823, 3.6845, 3.6875, 3.6932, 3.6953, 3.6964, 3.6997, 3.7083, 3.7105, 3.7115, 3.7153, 3.7237, 3.7262, 3.7321, 3.7375, 3.75, 3.7552, 3.7562, 3.775, 3.7798, 3.7813, 3.7833, 3.7841, 3.7933, 3.7969, 3.8108, 3.8125, 3.8167, 3.8203, 3.8235, 3.8295, 3.8323, 3.8333, 3.8438, 3.851, 3.8558, 3.8668, 3.8687, 3.87, 3.875, 3.89, 3.8906, 3.9, 3.9032, 3.9063, 3.9076, 3.9156, 3.9224, 3.925, 3.9352, 3.9408, 3.9583, 3.9722, 3.9766, 3.9886, 3.994, 4.0, 4.0027, 4.0048, 4.0061, 4.0122, 4.0125, 4.0132, 4.015, 4.016, 4.0227, 4.025, 4.0368, 4.0417, 4.0461, 4.0474, 4.0521, 4.0524, 4.056, 4.0625, 4.0717, 4.0764, 4.0798, 4.0833, 4.087, 4.088, 4.0972, 4.1, 4.1023, 4.1029, 4.1125, 4.1179, 4.125, 4.1295, 4.1328, 4.1375, 4.1379, 4.1442, 4.1467, 4.1513, 4.1635, 4.175, 4.1838, 4.1843, 4.1845, 4.1875, 4.1917, 4.2059, 4.2096, 4.21, 4.2132, 4.2312, 4.2344, 4.2383, 4.25, 4.2538, 4.26, 4.263, 4.2639, 4.2647, 4.2672, 4.275, 4.2788, 4.2813, 4.2826, 4.2891, 4.2917, 4.2981, 4.3, 4.3036, 4.3393, 4.3419, 4.3571, 4.3698, 4.375, 4.3882, 4.3889, 4.3897, 4.4083, 4.4091, 4.412, 4.4122, 4.4167, 4.4211, 4.4233, 4.4271, 4.433, 4.4357, 4.4375, 4.4529, 4.4531, 4.4567, 4.4583, 4.4655, 4.4659, 4.475, 4.485, 4.4896, 4.51, 4.5122, 4.5268, 4.5306, 4.5357, 4.5382, 4.5417, 4.5458, 4.5486, 4.5493, 4.5523, 4.5606, 4.5625, 4.5659, 4.5694, 4.5714, 4.5726, 4.5757, 4.5815, 4.5833, 4.5918, 4.6028, 4.6146, 4.6172, 4.6211, 4.625, 4.6467, 4.6556, 4.6563, 4.6607, 4.675, 4.6806, 4.6898, 4.69, 4.6908, 4.7083, 4.7097, 4.7143, 4.7188, 4.7325, 4.7356, 4.7468, 4.7679, 4.7708, 4.7891, 4.7986, 4.8047, 4.8125, 4.8173, 4.825, 4.8283, 4.8375, 4.8831, 4.8906, 4.9, 4.9196, 4.9219, 4.9286, 4.9312, 4.9485, 4.9589, 4.9662, 4.9688, 4.9803, 4.9844, 4.9904, 5.0, 5.0081, 5.0275, 5.0329, 5.0357, 5.036, 5.0406, 5.0532, 5.0605, 5.0622, 5.0672, 5.0754, 5.0762, 5.0795, 5.0827, 5.0838, 5.086, 5.0871, 5.0912, 5.0923, 5.0947, 5.0953, 5.0969, 5.1021, 5.1169, 5.1433, 5.155, 5.1668, 5.1701, 5.1738, 5.1788, 5.1896, 5.2375, 5.2548, 5.2639, 5.2671, 5.273, 5.3096, 5.3283, 5.3423, 5.35, 5.3649, 5.3727, 5.3749, 5.375, 5.3777, 5.3799, 5.3819, 5.4165, 5.4236, 5.4386, 5.4496, 5.4508, 5.4591, 5.4836, 5.5111, 5.5124, 5.5581, 5.5596, 5.5629, 5.5632, 5.5649, 5.5756, 5.5842, 5.5951, 5.6029, 5.6091, 5.6112, 5.6235, 5.6254, 5.6263, 5.6271, 5.6305, 5.6349, 5.636, 5.6533, 5.6549, 5.6652, 5.6704, 5.6935, 5.6973, 5.7251, 5.756, 5.7765, 5.785, 5.7934, 5.8171, 5.826, 5.8322, 5.8543, 5.8548, 5.8596, 5.8625, 5.8652, 5.8747, 5.8752, 5.8788, 5.9175, 5.9199, 5.9254, 5.9277, 5.9596, 5.9629, 5.9919, 6.0062, 6.0132, 6.0162, 6.0196, 6.04, 6.065, 6.0667, 6.07, 6.0749, 6.1253, 6.1632, 6.1943, 6.2048, 6.213, 6.2275, 6.2475, 6.2583, 6.2712, 6.3256, 6.3303, 6.3427, 6.3699, 6.3861, 6.4232, 6.4333, 6.4469, 6.4504, 6.48, 6.4963, 6.5162, 6.5195, 6.5319, 6.5356, 6.5853, 6.6086, 6.6188, 6.6369, 6.7058, 6.7099, 6.7247, 6.7703, 6.8112, 6.8336, 6.8469, 6.9296, 6.9454, 6.9737, 6.9909, 7.0215, 7.0352, 7.0549, 7.0842, 7.1069, 7.1267, 7.1381, 7.1831, 7.2392, 7.2419, 7.266, 7.3151, 7.4029, 7.5, 7.52, 7.5674, 7.598, 7.6344, 7.6348, 7.6775, 7.6864, 7.6886, 7.6986, 7.7116, 7.7197, 7.8627, 7.8647, 7.866, 7.8772, 8.0957, 8.1001, 8.1093, 8.2069, 8.3209, 8.3637, 8.3695, 8.4331, 8.4546, 8.5325, 8.603, 8.634, 8.72, 8.7477, 8.7974, 8.8178, 8.8636, 9.0776, 9.1531, 9.4356, 9.4664, 9.7646, 10.1007, 10.2264, 10.5144, 10.5793, 10.5941, 10.8758, 10.9405, 10.9506, 11.2866, 12.0088, 12.8763, 15.0001]

xticks = np.arange(0, 16, 5)
fig, ax1 = plt.subplots(figsize=(4, 4))
ax1.set_xticks(xticks, xticks, fontsize=10)

ax1.plot(xs, ys)

#yticks = np.arange(0, -0.81, -0.2)
#yticks = [round(y, 1) for y in yticks]
#ax1.set_yticks(yticks, labels=yticks, fontsize=12)

ax1.set_xlabel("Median Income", fontsize=13)
ax1.set_ylabel("Activation magnitude", fontsize=13)

plt.subplots_adjust(left=0.2, right=0.98, top=0.94, bottom=0.15)
plt.title("CalHousing")
plt.show()
