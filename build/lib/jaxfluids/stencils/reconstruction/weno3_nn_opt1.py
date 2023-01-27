#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from functools import partial
from typing import List

import haiku as hk
import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO3NNOPT1(SpatialReconstruction):
    """WENO3NNOPT1

    Bezgin et al. - 2021 - 
    """
    
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO3NNOPT1, self).__init__(nh=nh, inactive_axis=inactive_axis)
        
        self.dr_ = [
            [1/3, 2/3],
            [2/3, 1/3],
        ]
        self.cr_ = [
            [[-0.5, 1.5], [0.5, 0.5]],
            [[-0.5, 1.5], [0.5, 0.5]],
        ]

        self._c_eno = 2e-4

        self._stencil_size = 4

        self._slices = [
            [
                [   jnp.s_[:, self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[:, self.n-1+j:-self.n+j,   self.nhy, self.nhz],  
                    jnp.s_[:, self.n+j:-self.n+1+j,   self.nhy, self.nhz], ],  

                [   jnp.s_[:, self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[:, self.nhx, self.n-1+j:-self.n+j,   self.nhz],  
                    jnp.s_[:, self.nhx, self.n+j:-self.n+1+j,   self.nhz], ],   

                [   jnp.s_[:, self.nhx, self.nhy, self.n-2+j:-self.n-1+j,],  
                    jnp.s_[:, self.nhx, self.nhy, self.n-1+j:-self.n+j,  ],  
                    jnp.s_[:, self.nhx, self.nhy, self.n+j:-self.n+1+j,  ], ],

            ] for j in range(2)]

        self._get_nn()

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [   jnp.s_[..., 0, None:None, None:None],  
                    jnp.s_[..., 1, None:None, None:None],  
                    jnp.s_[..., 2, None:None, None:None], ],   

                [   jnp.s_[..., None:None, 0, None:None],  
                    jnp.s_[..., None:None, 1, None:None],  
                    jnp.s_[..., None:None, 2, None:None], ],

                [   jnp.s_[..., None:None, None:None, 0],  
                    jnp.s_[..., None:None, None:None, 1],  
                    jnp.s_[..., None:None, None:None, 2], ],
            ],

            [
                [   jnp.s_[..., 3, None:None, None:None],  
                    jnp.s_[..., 2, None:None, None:None],  
                    jnp.s_[..., 1, None:None, None:None], ],   

                [   jnp.s_[..., None:None, 3, None:None],  
                    jnp.s_[..., None:None, 2, None:None],  
                    jnp.s_[..., None:None, 1, None:None], ],

                [   jnp.s_[..., None:None, None:None, 3],  
                    jnp.s_[..., None:None, None:None, 2],  
                    jnp.s_[..., None:None, None:None, 1], ],
            ],
        ]

    def reconstruct_xi(self, buffer: jnp.DeviceArray, axis: int, j: int, dx: float = None, **kwargs) -> jnp.DeviceArray:
        s1_ = self._slices[j][axis]

        dx1 = jnp.abs( buffer[s1_[1]] - buffer[s1_[0]] )
        dx2 = jnp.abs( buffer[s1_[2]] - buffer[s1_[1]] )
        dx3 = jnp.abs( buffer[s1_[2]] - buffer[s1_[0]] )
        dx4 = jnp.abs( buffer[s1_[0]] - 2*buffer[s1_[1]] + buffer[s1_[2]] )

        x = jnp.stack([dx1, dx2, dx3, dx4], axis=-1) 
        x /= (jnp.maximum(x[:,:,:,:,:1], x[:,:,:,:,1:2]) + self.eps)

        omega_z_ = self.net.apply(self.params, x)
        omega_z_ = jax.nn.relu(omega_z_ - self._c_eno)
        omega_z_ /= jnp.sum(omega_z_, axis=-1, keepdims=True)

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]] 
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]]

        cell_state_xi_j = omega_z_[:,:,:,:,1] * p_0 + omega_z_[:,:,:,:,0] * p_1

        return cell_state_xi_j

    def _get_nn(self) -> None:
        w = jnp.array([
                [-3.3636199430413627, -0.7351760486626531, -1.0380715848953586, -0.10683684612079541],
                [0.6320153775249236, -0.25063347973585137, 0.9391343557951499, -0.5485699790719365],
                [0.48572636212321973, -0.19663348669125083, 0.593751506792153, 0.37210723908968374],
                [0.30279189440268434, -0.5747518777720824, -0.18765553236960378, 0.5617584946856855],
                [-0.9698969495831152, 0.3886446134687482, -0.128269159726216, 0.2593053476193478],
                [0.5574079110033463, -0.9599209486951579, 0.02395137931149818, 0.05355918238361052],
                [0.8102211002151866, 0.3799165797496963, 0.7634586940042873, -1.122915300744421],
                [0.8185890553975251, 0.24799345538559422, 0.47720309136792444, 0.02614474430945657],
                [0.5654335001434448, -1.225339449727529, -0.0445224691137319, -0.057350544935499556],
                [-1.037315695984657, 0.5506128559787825, -0.18710096179291355, 0.44242980846989893],
                [-0.3047941181045883, 0.21408484021935992, 0.3216093649696904, -0.06647501451265608],
                [-0.5784548672177117, -0.525901002632359, -0.40231258078718646, -0.514536341833738],
                [0.6885144337241698, 0.07727692424146439, -0.002517270427443718, 0.12648842758609685],
                [-0.5141463730415861, 0.6308603138360143, -0.18660424435683695, 0.7089138604963139],
                [0.08242892614298822, -0.7327836826685112, 0.3262208988224903, 0.2906869687847254],
                [0.28374813431860063, -0.36484459817691434, 0.024480049936531603, 0.5142434385581227],
            ], dtype=jnp.float64).transpose()

        w1 = jnp.array([
            [-0.5781986969868682, -0.44393846962070416, -0.39001715876042886, 0.15156837812597834, 0.5523831427370872, 0.08234243151891239, -0.19660180924651052, 0.04858237091898023, 0.20997633375361266, 0.8082310225107752, 0.2910698992652692, -0.27559680454580876, -0.37582930442611656, 0.6414809089603271, -0.10161600751871838, 0.12462173639634255],
            [-0.39227854973151793, -0.1918000966649267, 0.19432514394815054, 0.36070415015106105, -0.4233795233411991, 0.5057178756681626, -0.5082012057887959, -0.36011441650327125, -0.05787874209449309, -0.20720192703198212, -0.39847677235058293, -0.12093537821090297, -0.30257795904161094, -0.45580621657790105, 0.5854224930452867, 0.4312881666082077],
            [-0.08399129667126241, 0.7204755063035373, 0.2642161852952745, 0.05356348026512046, -0.48752393497195906, 0.30696390499747456, 0.4110563728472665, 0.6468482798331587, -0.23285755642187478, -0.2960279705133975, 0.13322513579967582, 0.10992804995050258, 0.329363785487234, -0.16360705604874587, -0.06909576905127189, -0.1005487783484379],
            [-0.38177986948152814, 0.5999826900783087, 0.2183569724742987, 0.15781690480489577, -0.17390355857547576, 0.31242899658625783, 0.2958823791998624, 0.41593263428031857, -0.2581463621018953, -0.019593523658324175, 0.08318999089346635, -0.5097292078190624, 0.2214382243699303, -0.2971695892224367, 0.2965797033176569, 0.1195967034093866],
            [-0.45716982517830773, -0.341113377170982, 0.25459599721595, -0.18627777067840412, 0.2358387265604964, -0.07224043620241215, 0.10284502498191694, -0.5122070791509102, -0.41203733660561054, 0.44583945582143847, -0.189311685363244, -0.10107807871597048, 0.32357783898839254, 0.2785610843782067, 0.1810028252930368, -0.124881394414201],
            [-0.0353037067346609, 0.14156000575486577, 0.22809796243619557, -0.11505112548996929, 0.01240625154618994, -0.3301017091488843, -0.6095770411385663, 0.3253977917998842, -0.48011173168384336, 0.7428002308535165, 0.5141368386034507, -0.4494603644026563, -0.2211206684351633, 0.03214893053827113, -0.10423509275225028, -0.15668608366740377],
            [-0.19820194376319525, -1.0796110775135772, 0.6601722495731629, 2.16155152333764, -0.6272726174953362, 0.4771276707641084, -2.786815212210548, 0.11380838404166727, 0.774705776618212, 0.2240127981655552, -0.28521239015789807, -0.28379425058820906, 0.009332415882640105, 0.7549354868171777, 0.3278654758928927, 1.8689216230445014],
            [-0.5673592867337381, -0.5870289638071496, -0.28760990140325143, -0.2599390593957131, 0.7333289409970646, 0.017882429207095586, -0.5140847740857992, -0.3724027155240346, -0.45829486022473814, 0.9380880379918737, -0.005436870947663284, 0.10347460955583197, -0.43524837376968234, 0.6450218331638167, -0.11393420240350285, -0.5325005110481795],
            [-0.1578167281346622, -0.4990072663887819, 0.16061424981935077, -0.11930332701526178, 0.23879871652365614, -0.3728781843574354, -0.6482446979009322, -0.46328301844968905, -0.2510903891750089, 0.09479182227436657, 0.21963357463275465, -0.3986999441843772, -0.13177535742921062, 0.23867835492583583, 0.31998838765086923, -0.35186289223678235],
            [-0.23628558923758308, -0.2927580930402581, 0.46194674374290445, 0.1008639903490192, -0.40247250944021706, 0.2627736314469531, 0.15940045835220074, 0.5062559263751959, 0.4916095242072076, 0.1876385551979399, 0.136008414916649, -0.058497968208373975, -0.03413857101302722, -0.37589354033821465, -0.16538151313499982, -0.26290759603192415],
            [-0.4884809314198987, -0.5073030589542719, -0.4490378987182159, -0.23273730858527866, 0.6893768087707252, -0.04388510795013468, 0.042476460407702005, -0.27972192333461743, -0.3677331941407762, 0.11595050664917898, 0.4719638957470959, 0.07589890845910985, 0.15642530187537587, 0.6168817045308265, 0.19778089011940284, 0.0108163613627861],
            [0.1085494595082266, -0.4259569139134789, -0.3588220940964178, 0.09827255616441233, -0.2109427038074598, 0.1921765596648278, 0.19033789382911273, -0.4254009926990324, 0.14840230873184765, -0.0918427678660932, 0.2106647515268193, 0.28514101952346144, 0.31248571876255943, 0.21192684985220675, -0.018668804356491115, -0.08496544417057687],
            [0.19326764320398732, 0.17832736265067736, 0.019234758905393067, 0.06140130771238733, -0.15489571751873152, -0.07855547735747531, -0.01775747135144814, -0.03403266791260739, 0.16038211452159334, -0.2509163806863737, -0.048558917194135943, -0.09124566191338812, -0.3699442484162329, 0.29123811432696783, -0.21187610777897023, -0.3339325710077664],
            [0.23024531152705982, -0.22397400313577948, 0.4061273989699611, -0.20156610952835874, 0.04414084889252632, 0.42298501033862435, -0.05075944138678446, 0.13202879275565935, 0.13782982633720106, -0.49114424688552344, -0.18090842462474338, 0.24531095418745616, 0.30092951828382397, 0.1481427745164258, 0.08541398353765514, 0.17416186757007607],
            [0.25839918655426075, 0.033529882774553654, 0.1958799501134463, 0.2667829730791037, -0.4279307261642568, -0.20778673894258295, 0.020110322946891204, 0.09106997980365195, 0.09180439878510975, -0.43247608905018603, 0.2713787812997134, -0.23250192561918043, -0.36603026790450405, -0.18651883959370552, 0.3076164181940168, -0.03699973230679113],
            [-0.4451855535736983, -1.3243173337346472, 0.4241481537039716, 0.30899289824381876, 0.3836547649789459, 0.34770207876696324, -1.2155438040097826, 0.1465518196648991, 0.06244619664849472, 1.3558040814537398, -0.1411293164081895, -0.5841960373619205, -0.3111101226240681, 0.4932475298289673, 0.33155955230178613, 0.9777725670116078],
            ], dtype=jnp.float64).transpose()

        w2 = jnp.array([
            [0.10107902072398287, -0.025850661059006134, -0.19405547248473773, 0.2922344182722797, 0.0954835967615693, 0.14971467819759163, 0.06273866656512406, 0.05919403856321677, 0.2630383059121237, -0.25573502654448665, -0.22771517316143927, 0.09028140908102353, -0.26023294768832844, 0.13859784198621147, 0.35437595293605156, -0.1948033337523156],
            [0.25474606013323137, -0.1743718887032992, -0.007806759477912259, 0.13146786895513518, -0.1787785506275865, -0.14397681256558428, -0.13372479795797798, -0.055257674739710376, 0.0663616094328213, -0.12828240106312938, 0.12993065398275586, 0.08818574812732052, 0.19901560901730792, -0.27048213282495526, 0.21843981810881444, 0.1036569832262954],
            [-0.20949870960605815, -0.053613311238269394, -0.368186672429691, 0.11200006869443371, 0.0922693135202921, 0.055838300176904256, 0.42076818248035597, -0.2886726484207812, 0.25273040495394183, 0.11864780556264957, 0.09802992187170148, -0.2237911421534205, -0.3619321832783275, 0.10822697233439012, 0.3510368386382138, 0.28866548953140464],
            [-0.05650583957087169, 0.03153085438627896, -0.07952512989706996, 0.36179683554733366, 0.1558792745453362, 0.12201450378231803, 0.003199518521492076, 0.03253298726923804, -0.1392173001874409, -0.23400782002191511, -0.3607436556297695, 0.222997865774028, -0.09896245359439204, -0.35357616901515954, 0.25953541437515015, -0.17668446070975405],
            [0.8885418867273566, 0.001327460394526184, -0.26787504407832513, -0.34546981672553595, 0.4970075829476536, 0.25228687598351107, 0.18535316299631793, 0.6900024446230255, 0.22909186354008929, -0.5929006071611729, 0.3185735373272068, 0.11761713945502322, -0.26575517384022385, 0.1464566184642636, -0.04165502554664985, 0.3398173573707204],
            [0.7226023405806071, 0.1300425269037079, -0.9121063910325242, -0.5934683678485366, 0.06638493665119428, 0.1423103230868066, -0.4525514916823224, 0.31498793553593, 0.5514384433510405, -0.5700652564109139, 0.647001823324452, 0.3632049164080918, 0.29792156221101773, -0.5107487413816312, -0.5184178168250595, 0.35747441764931004],
            [-0.44671712848797657, 0.07193460095476585, 0.24446175691468855, 0.3363475040109809, -0.2558031952091493, -0.05762937090988027, -0.44356040676174774, -0.3650527896338257, -0.5999828844526301, -0.10733829981201555, -0.2509396840066568, -0.039869144866636765, 0.21916939290216533, -0.24298354245026962, 0.35657134082456393, -0.8901912818685964],
            [0.022738673528511125, 0.27897282322910705, -0.04488614894444226, -0.01760735046486867, -0.2916819193075569, -0.26762878122196415, 0.8259948988156169, -0.2770771342236844, -0.21636295617449808, -0.08782523280371353, -0.2566676260311571, -0.08409555236998674, -0.11031856346177855, -0.299711637242986, 0.062187990267165656, 0.5366284432058411],
            [0.17029014523834496, -0.09674642531312941, -0.5471915771838345, -0.518409539721132, 0.45959238322166546, 0.20573255197486903, 0.41021907843174243, 0.5180459396059507, -0.21009871872403965, 0.27367689893488734, 0.5335390378970684, -0.25575339071497866, -0.13325289963421805, 0.24427537876119212, 0.21094959876511082, 0.16944071686432932],
            [0.5608898219862919, -0.33144031811250585, -0.44386023552552223, -0.2995024538185788, 0.25473739212034885, -0.026747238923908363, 0.1313402525265284, 0.32688331634822043, -0.2138430175504173, -0.2312906537509162, 0.17598152715437446, 0.33911655108944716, -0.31877297964246265, 0.20973601832811398, 0.04493023841710308, 0.47279958926039894],
            [0.5947674316338799, -0.3646559616946315, 0.22042680635361558, -0.34365091603277254, -0.11083434241788549, 0.39319480185064787, -0.049140895652295326, 0.35224754986401596, 0.10602824327772471, -0.36136580610930474, -0.15149804394046582, 0.5853236341760505, -0.2330247393249639, -0.05212960042336549, -0.2493753019947928, 0.3754215218870083],
            [0.35676455782106054, 0.31353635978557115, -0.5121744075485136, -0.34156440031805, 0.5396917430490541, 0.18668672120207783, 0.38085462903983186, 0.43962562207097655, 0.6101360083106929, -0.24458750088228043, 0.4345810915322202, 0.1304419895170894, 0.3655203142653417, -0.17947344061851808, -0.5167718371808433, 0.6336747672583105],
            [0.5951036798668391, -0.3200751498881675, -0.30023281957976755, 0.22251319432646266, -0.33168843916448376, 0.35021316171837386, 0.38056033994776073, 0.1956862596245976, 0.029511313409638192, -0.36104823084850074, -0.19785646615113764, -0.3210770877443572, -0.259833187026932, -0.008180969795341903, 0.041921731629268745, 0.008660364778113907],
            [-0.517735941702895, 0.857592995559161, 0.32591607364772535, -0.24039780434085245, 0.025391663463268026, -1.1799827163787837, 2.295260190081591, -0.1402867986675579, -0.17284930792390737, -0.37596601010477054, -0.10937208013455932, 0.8138959493205707, 0.011815365368139315, 0.47054976181912955, 0.04145804332504425, 0.9514503790086146],
            [-0.05625799671142728, 0.4931456999601273, -0.20739411312393297, 0.3961685711325033, -0.35748588542845117, 0.33743004446200364, 0.2154331616428584, -0.2859370541890416, 0.1520175960638835, 0.49447802507088684, -0.33938185690025374, -0.386530692935652, 0.23277654269477238, 0.4369887952185995, -0.29370244489599817, 0.3810600413958437],
            [0.2122136531376884, -0.521280808051434, -0.23434035217123653, 0.3089933239762192, -0.2798417931219857, 0.47436093089469145, -0.020897409829692883, 0.42394050447549253, 0.27078361216362673, 0.087578089533156, 0.26293069222494136, -0.20901797787019646, -0.17873224199963714, 0.4534117138117518, -0.2790851120445561, 0.39986781320801096],
            ], dtype=jnp.float64).transpose()
        
        w3 = jnp.array([
            [-0.20469325596820687, -0.34115257543583677, 0.19728763175730454, 0.2063227740634609, 0.26773686369505223, -0.7620165039161816, 0.28886161355374906, 0.4639857562556888, -0.35162280785200667, -0.321144798222583, -0.442719130872063, -0.4545268173759951, -0.5967498343146402, 0.9262972364700224, 0.5874340560892706, -0.5937066592454496],
            [-0.20946972651791432, -0.3746953074291135, -0.5775919167836157, -0.1287699134905062, 0.7070379766590361, -0.11248845371901833, 0.5481818017059644, -0.5709465078343058, 0.5227467274541583, 0.622767210502863, 0.18263193635644256, 0.8738176370255476, 0.6373937276993936, 0.06939171136743856, -0.17971755236635154, 0.03370083344948084],
            ], dtype=jnp.float64).transpose()

        b = jnp.array([
            -0.10388973620646459, 0.28569364904241784, 0.14218029836430307, 0.2358534653299724, 0.13215845583055655, 0.13202588947896907, 0.21907165887779737, 0.2686640601861153,
            0.1144287254972733, 0.2578297565037551, 0.02891001147862591, -0.12100540073040143, -0.09763531217743227, 0.05773908364022649, 0.06674230971111526, 0.09717028497781278
            ], dtype=jnp.float64)

        b1 = jnp.array([
            0.022476759818808884, 0.06522787447191643, -0.014704387279203337, 0.1096374663689587, -0.05440171127270248, 0.09028210024793715, 0.32331745875105944, 0.057105461894815614,
            0.057290296437207955, 0.16575651814670797, 0.028858095485021478, -0.23377543412696206, 0.004647878425389973, 0.059561098550835405, 0.1043107038091163, 0.17370218033430374
            ], dtype=jnp.float64)

        b2 = jnp.array([
            0.004819399003695486, -0.024907489079537354, 0.06724733986081394, 0.01979302017757484, -0.04114135845688159, -0.011615947262669913, -0.024769012012465872, 0.07458398675892401,
            -0.01389492808996807, -0.017278122108917517, -0.010393727683392022, -0.0022579707865170583, -0.020020447735082682, -0.25852216119792715, 0.09445572505091969, -0.029852434851929288
            ], dtype=jnp.float64)

        b3 = jnp.array([
            0.03510259928999757, -0.03510259928999723
            ], dtype=jnp.float64)

        params = {
            'linear': {'w': w, 'b': b,},
            'linear_1': {'w': w1, 'b': b1,},
            'linear_2': {'w': w2, 'b': b2,},
            'linear_3': {'w': w3, 'b': b3,},
        }

        self.params = hk.data_structures.to_immutable_dict(params)

        @jax.jit
        def net_fn(x):
            mlp = hk.Sequential([
                hk.Linear(16), jax.nn.swish,
                hk.Linear(16), jax.nn.swish,
                hk.Linear(16), jax.nn.swish,
                hk.Linear(2), jax.nn.softmax,
            ])
            return mlp(x)

        self.net = hk.without_apply_rng(hk.transform(net_fn))