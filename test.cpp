    // Conv2D* c = Conv2D::Create(8,8,1, 3,3,1);
    // f32 k[] = {
    //      0.01811075,
    //     -0.22047621,
    //      0.39133304,
    //    -0.45425707,
    //     -0.12132424,
    //     -0.36175254,
    //    -0.00107366,
    //      0.5620297,
    //      0.06265229
    // };
    // c->kernels = M4(k, c->dkernels.d1, c->dkernels.d2, c->dkernels.d3, c->dkernels.d4);
    // MaxPooling* p = MaxPooling::create(c->getOutputSize().h,c->getOutputSize().w, c->getInputSize().c, 2,2,1);
    // Layer* l = Layer::create(p->getLinearFlattenedSize(), 10);
    // f32 w[] = {
    //     -0.19917646, -0.34107822, -0.24724421,  0.20609385,  0.18097705,
    //     -0.17875314, -0.19397977, -0.05015582, -0.31834078, -0.20006394,
    //     0.4549566 , -0.49765623, -0.2638585 , -0.43737757, -0.39030218,
    //      0.42843938, -0.48435795,  0.2547267 , -0.38572627,  0.40745628,
    //    -0.24073252,  0.44760674, -0.21860966, -0.4141187 , -0.21736005,
    //      0.01641119, -0.23804864,  0.12451601, -0.41855103,  0.24135596,
    //    -0.08290648,  0.32980728,  0.2688132 , -0.56192786,  0.10637796,
    //      0.12181979,  0.33724022, -0.53835696, -0.35881832, -0.2226134 ,
    //    -0.27798408, -0.17728338,  0.43555474, -0.42483398, -0.06410003,
    //     -0.28865117,  0.33679742, -0.5491426 , -0.5143939 , -0.39436686,
    //     0.34707797,  0.12405902,  0.1748972 ,  0.16989213,  0.10169607,
    //     -0.388882  ,  0.14807487, -0.25465208,  0.00214475, -0.21123582,
    //    -0.4591869 , -0.12115058, -0.24839911, -0.33392918, -0.55885357,
    //      0.02472621,  0.48376566,  0.5082317 ,  0.06114483,  0.39543605,
    //     0.31763995, -0.2784083 , -0.48701555,  0.43811935,  0.432768  ,
    //      0.12962562,  0.17145085,  0.0680173 ,  0.04852182, -0.14138621,
    //    -0.26183525, -0.47246736,  0.10499758, -0.39079833, -0.12245113,
    //      0.313542  , -0.525171  ,  0.55781716, -0.38507313, -0.03552389
    // };

    // f32 w2[] = {
    //     -1.28007650e-01, -2.19205722e-01, -1.58900052e-01,
    //      1.32453322e-01,  1.16311163e-01, -1.14881888e-01,
    //     -1.24667823e-01, -3.22343707e-02, -2.04592735e-01,
    //     -1.28578022e-01,
    //     2.92393655e-01, -3.19836020e-01, -1.69577792e-01,
    //     -2.81095833e-01, -2.50841230e-01,  2.75351375e-01,
    //     -3.11289400e-01,  1.63708955e-01, -2.47900337e-01,
    //      2.61865884e-01,
    //    -1.54715091e-01,  2.87669986e-01, -1.40497074e-01,
    //     -2.66147733e-01, -1.39693961e-01,  1.05472207e-02,
    //     -1.52990207e-01,  8.00245404e-02, -2.68996298e-01,
    //      1.55115753e-01,
    //    -5.32827377e-02,  2.11962074e-01,  1.72762126e-01,
    //     -3.61142397e-01,  6.83674812e-02,  7.82917142e-02,
    //      2.16739088e-01, -3.45993757e-01, -2.30607018e-01,
    //     -1.43070206e-01,
    //    -1.78656101e-01, -1.13937303e-01,  2.79924363e-01,
    //     -2.73034275e-01, -4.11961079e-02, -1.85511664e-01,
    //      2.16454536e-01, -3.52925509e-01, -3.30593050e-01,
    //     -2.53453523e-01,
    //     2.23061651e-01,  7.97308087e-02,  1.12403750e-01,
    //      1.09187037e-01,  6.53584898e-02, -2.49928504e-01,
    //      9.51654315e-02, -1.63660973e-01,  1.37838721e-03,
    //     -1.35758027e-01,
    //    -2.95112371e-01, -7.78616071e-02, -1.59642294e-01,
    //     -2.14611158e-01, -3.59166622e-01,  1.58911645e-02,
    //      3.10908705e-01,  3.26632649e-01,  3.92968357e-02,
    //      2.54140645e-01,
    //     2.04142302e-01, -1.78928718e-01, -3.12997401e-01,
    //      2.81572610e-01,  2.78133363e-01,  8.33083689e-02,
    //      1.10188812e-01,  4.37136889e-02,  3.11842263e-02,
    //     -9.08667445e-02,
    //    -1.68277502e-01, -3.03647518e-01,  6.74803257e-02,
    //     -2.51160085e-01, -7.86974430e-02,  2.01508671e-01,
    //     -3.37519318e-01,  3.58500510e-01, -2.47480571e-01,
    //     -2.28306651e-02,
    //    -2.75716245e-01, -2.21218035e-01, -2.13938236e-01,
    //      2.88266450e-01,  3.38028103e-01, -1.04075402e-01,
    //     -1.96368098e-02,  2.58489639e-01, -3.07655275e-01,
    //      9.20528471e-02,
    //    -9.69347358e-03, -7.50808716e-02,  1.44541651e-01,
    //     -2.86098480e-01, -1.49218127e-01, -8.03698897e-02,
    //     -1.88787326e-01,  2.00558394e-01,  6.03478551e-02,
    //      1.29476190e-01,
    //    -1.54030368e-01, -1.12082049e-01,  1.49244756e-01,
    //     -1.36355251e-01,  9.78658199e-02,  1.49488449e-04,
    //     -2.36767530e-01, -1.65234655e-01,  5.29852211e-02,
    //      3.56810361e-01,
    //    -3.47554684e-01, -6.03264272e-02, -1.87135026e-01,
    //     -2.47740179e-01, -2.14836895e-02,  3.42378646e-01,
    //      8.13631415e-02, -3.56429368e-01, -2.00764686e-01,
    //      6.13320470e-02,
    //     1.04408145e-01,  2.55502194e-01, -1.07804447e-01,
    //      3.10725242e-01, -1.91916451e-01, -8.71485472e-03,
    //     -3.16003382e-01, -1.58253610e-02,  3.26265186e-01,
    //      2.96122432e-02,
    //     3.01327735e-01, -9.23148692e-02, -6.46678209e-02,
    //     -7.56520331e-02, -2.03847289e-02, -3.13718200e-01,
    //      3.03086609e-01,  1.97797209e-01, -1.93804085e-01,
    //      1.12495720e-02,
    //     5.39077818e-02, -2.36931890e-01, -3.05149138e-01,
    //      4.35426831e-02, -2.70862490e-01, -2.51354516e-01,
    //      3.33012074e-01, -8.26079845e-02,  8.91436636e-02,
    //     -2.21777290e-01,
    //    -2.49126673e-01, -1.46749020e-01, -1.05434537e-01,
    //      3.52074802e-02, -6.72289133e-02, -3.37643743e-01,
    //     -5.17170429e-02, -2.91535556e-02,  1.61100775e-01,
    //      2.71005660e-01,
    //    -1.02996588e-01, -1.67282104e-01,  3.47996444e-01,
    //     -1.96794301e-01, -3.43312562e-01, -1.59927472e-01,
    //     -3.16172510e-01,  1.92128807e-01,  1.48906082e-01,
    //     -3.36319238e-01,
    //    -1.86794400e-01, -2.15406954e-01,  2.69073278e-01,
    //     -2.58115441e-01,  3.42212349e-01, -3.52989137e-01,
    //      3.54293257e-01, -3.10369253e-01,  1.85572952e-01,
    //     -3.36662024e-01,
    //     3.69140208e-02, -2.60214031e-01,  2.03639895e-01,
    //     -3.60198259e-01,  3.36535722e-01,  3.32322150e-01,
    //     -1.82839632e-02, -3.20235461e-01, -1.93757758e-01,
    //      7.33341277e-02,
    //    -1.08160317e-01, -1.12043470e-01,  4.68880832e-02,
    //     -2.32582048e-01, -2.73691446e-01,  2.84924597e-01,
    //      1.60495073e-01, -2.17559442e-01, -6.50054216e-03,
    //     -1.22247964e-01,
    //     1.50668412e-01,  1.43030673e-01, -4.66302931e-02,
    //      6.06492460e-02,  1.86451763e-01,  2.85112292e-01,
    //     -1.85821384e-01, -2.49671981e-01,  2.52585739e-01,
    //     -9.28188562e-02,
    //    -7.84060359e-03,  2.86419660e-01, -2.15958461e-01,
    //      3.01613063e-01, -1.91565216e-01,  2.91752070e-01,
    //      3.04199487e-01,  3.09268206e-01,  3.23797852e-01,
    //     -5.75291514e-02,
    //     1.16702110e-01,  1.41206950e-01, -5.44591248e-02,
    //      3.54511827e-01, -1.12548485e-01, -2.89758444e-02,
    //     -3.28590482e-01, -4.06547487e-02, -2.36206800e-01,
    //      2.05445558e-01,
    //     1.83267266e-01,  1.50170922e-03,  5.17060161e-02,
    //     -2.95081466e-01, -2.23703593e-01,  4.61525619e-02,
    //      1.39128000e-01, -3.45264256e-01,  4.10832167e-02,
    //     -2.23176360e-01,
    //    -2.20267415e-01, -1.19994819e-01, -2.21863747e-01,
    //     -2.32635945e-01,  2.35431045e-01,  1.28552258e-01,
    //      3.05459887e-01, -3.34879160e-02, -3.03248167e-01,
    //      1.03690952e-01,
    //     6.32165074e-02,  1.42986685e-01, -2.60309875e-01,
    //     -1.54826939e-01,  1.15662277e-01,  1.11587375e-01,
    //      7.13889003e-02,  1.04045033e-01,  1.00925297e-01,
    //     -1.14605427e-02,
    //    -5.93900979e-02,  3.15415472e-01, -1.25802800e-01,
    //     -6.92835748e-02,  1.67497188e-01,  1.84068710e-01,
    //      1.83549196e-01, -3.17370176e-01, -1.78091332e-01,
    //     -1.23011559e-01,
    //    -1.83028281e-02,  2.00938433e-01,  2.09325999e-01,
    //      2.83277184e-01, -4.35381234e-02, -1.84271634e-02,
    //     -1.64203957e-01,  1.32638216e-03,  1.91881329e-01,
    //      2.47538239e-01,
    //     1.99561268e-01, -2.28212386e-01,  1.78729892e-02,
    //      1.53174132e-01, -2.48630524e-01, -2.23883376e-01,
    //      6.15604222e-02, -2.17538521e-01, -2.62711465e-01,
    //     -1.28364474e-01,
    //     2.48340160e-01,  1.93155676e-01, -8.82413983e-03,
    //      6.99301660e-02, -1.73217431e-01,  3.27492625e-01,
    //      2.28378385e-01,  5.08687198e-02,  1.10150069e-01,
    //      1.64311439e-01,
    //     1.72499120e-02, -3.18811327e-01, -2.24930272e-01,
    //     -2.90882707e-01,  8.56738091e-02,  1.95489615e-01,
    //     -3.46374691e-01, -1.30055085e-01, -1.57186523e-01,
    //      2.15712041e-01,
    //    -1.31649792e-01, -2.07648143e-01,  8.05363357e-02,
    //     -6.70616031e-02,  1.98740035e-01, -1.02656484e-02,
    //      2.33342201e-01, -2.47967839e-01,  2.53239274e-03,
    //     -1.48087978e-01,
    //     1.35485053e-01,  1.65851265e-01,  2.40571827e-01,
    //      1.75702304e-01, -9.34783518e-02,  1.12397462e-01,
    //      1.21777356e-02,  3.10046881e-01, -2.08033040e-01,
    //     -1.39893889e-01,
    //    -2.15140715e-01, -2.19157040e-02,  2.09922582e-01,
    //      2.74526566e-01,  1.56198114e-01,  1.31547391e-01,
    //      3.07956308e-01,  9.05997157e-02, -1.80852517e-01,
    //     -3.38567644e-01,
    //     2.77283192e-02,  1.28311425e-01,  2.95827121e-01,
    //      1.13335490e-01,  1.14861995e-01, -3.06742877e-01,
    //     -1.49937123e-01, -2.63098240e-01,  1.42169446e-01,
    //     -1.27706438e-01
    // };
    // l->w = M(w, l->w.rows, l->w.cols);

    // M3 a = Sigmoid(c->convolve2D(input[0]));
    // M3 b = p->forward(a);
    // M flatten = M(b.data, 1, b.d1*b.d2*b.d3);
    // M o = Softmax(l->forward(flatten));
    // M loss = M::MatMul(CrossEntropyPrime(y[0], o), SoftmaxPrime(o));
    // M d = l->backward(loss);

    // M3 dcnv = M3(d.data, b.d1, b.d2, b.d3); 
    // //assert(d.rows*d.cols == a.d1 * a.d2 * a.d3);
    // //M3 dcnv1 = M3(d.data, a.d1, a.d2, a.d3) * SigmoidPrime(a); 

    // p->backward(dcnv);
    // // u32 u=0;
    // // for(u32 i=0;i<p->d.d1;++i){
    // //     for (u32 j=0;j<p->d.d2;++j){
    // //         for(u32 k=0;k<p->d.d3;++k){
    // //             if(p->d(i,j,k) == 1.0f){
    // //                 p->d.set(i,j,k, d.data[u++]);
    // //             }
    // //         }
    // //     }
    // // }
    
    // M3 t = p->d * SigmoidPrime(a);
    // c->backward_conv(input[0], t);

    // l->dw = l->getDelta(loss, flatten);
    // l->db = loss;

    // // l->dw.print();
    // c->dkernels.print();
    // c->db.print();

    // return 0;