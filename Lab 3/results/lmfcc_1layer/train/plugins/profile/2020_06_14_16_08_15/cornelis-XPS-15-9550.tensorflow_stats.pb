"��
mHost_MklSoftmax"activation_2/Softmax(1�K7�A"�@9�K7�A"�@A�K7�A"�@I�K7�A"�@a#(���?i#(���?�Unknown
VHostIDLE"IDLE(1�MbX �@97�A`�js@A�MbX �@I7�A`�js@a6���U��?i>�e�G��?�Unknown
nHost_MklFusedMatMul"activation_1/Relu(1F���Զ�@9F���Զ�@AF���Զ�@IF���Զ�@aeG�fPg�?ix�����?�Unknown
lHost_MklFusedMatMul"dense_2/BiasAdd(1NbX9Br@9NbX9Br@ANbX9Br@INbX9Br@aj��5��?i)����?�Unknown
�HostSoftmaxCrossEntropyWithLogits"Qloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits(1V-��q@9V-��q@AV-��q@IV-��q@a�_�c�?i���+��?�Unknown
|Host
_MklMatMul"$gradients/dense_2/MatMul_grad/MatMul(1���Qn@9���Qn@A���Qn@I���Qn@a�P�d�?i	ڨ�k��?�Unknown
�Host_MklReluGrad")gradients/activation_1/Relu_grad/ReluGrad(1-����g@9-����g@A-����g@I-����g@a�*i?��?ii�;���?�Unknown
~Host
_MklMatMul"&gradients/dense_2/MatMul_grad/MatMul_1(1�V�\@9�V�\@A�V�\@I�V�\@aϗ��Z�?i�FPW��?�Unknown
k	HostArgMax"metrics/accuracy/ArgMax(1F�����A@9F�����A@AF�����A@IF�����A@a�X��/�?i������?�Unknown
m
HostArgMax"metrics/accuracy/ArgMax_1(1�A`�Ђ>@9�A`�Ђ>@A�A`�Ђ>@I�A`�Ђ>@a�KCL��{?i@0rԟ,�?�Unknown
�HostBiasAddGrad"*gradients/dense_1/BiasAdd_grad/BiasAddGrad(1D�l���=@9D�l���=@AD�l���=@ID�l���=@a���{?i��J�b�?�Unknown
~Host
_MklMatMul"&gradients/dense_1/MatMul_grad/MatMul_1(1�E����:@9�E����:@A�E����:@I�E����:@a�q�ƢNx?i��2�v��?�Unknown
WHostMul"mul_13(1���S��2@9���S��2@A���S��2@I���S��2@a���*3q?i��ܵ�?�Unknown
�HostMul"dgradients/loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mul(1�C�l��0@9�C�l��0@A�C�l��0@I�C�l��0@a�%�Gl�n?iB��Q���?�Unknown
XHostSqrt"Sqrt_3(1�&1�*@9�&1�*@A�&1�*@I�&1�*@a��$� �g?i2�R���?�Unknown
�HostBiasAddGrad"*gradients/dense_2/BiasAdd_grad/BiasAddGrad(1      &@9      &@A      &@I      &@a����d?i�ހ>� �?�Unknown
fHostGreaterEqual"GreaterEqual(15^�I�!@95^�I�!@A5^�I�!@I5^�I�!@a�����)`?i[_}˺�?�Unknown
[Host_MklMul"mul_12(1�Zd; @9�Zd; @A�Zd; @I�Zd; @a%K}��]?iˀ�?�Unknown
\Host	_MklAddV2"add_7(1}?5^�	 @9}?5^�	 @A}?5^�	 @I}?5^�	 @a�����1]?i���.�?�Unknown
pHostAssignVariableOp"AssignVariableOp_4(1�ʡE�s@9�ʡE�s@A�ʡE�s@I�ʡE�s@a�ѥ5�[?i�6�1�;�?�Unknown
WHostMul"mul_11(1����x�@9����x�@A����x�@I����x�@a`�>Wg�W?i4�>�G�?�Unknown
lHostMinimum"clip_by_value_3/Minimum(1ˡE���@9ˡE���@AˡE���@IˡE���@a����U?i6%��R�?�Unknown
WHostSub"sub_10(1�I+�@9�I+�@A�I+�@I�I+�@a:���iU?i���_]�?�Unknown
XHostAddV2"add_8(1F����x@9F����x@AF����x@IF����x@a4�Gr�\U?ij�FVh�?�Unknown
[Host_MklMul"mul_15(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@ac,�	T?i�3[r�?�Unknown
dHostMaximum"clip_by_value_3(1����K@9����K@A����K@I����K@a�b�aS?i�@BM�{�?�Unknown
\HostSquare"Square_2(1H�z�G@9H�z�G@AH�z�G@IH�z�G@a�<�*^S?i$Fbw��?�Unknown
^HostRealDiv"	truediv_3(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@a.��?=R?i~���?�Unknown
�HostConcatV2"Zloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1(1�����@9�����@A�����@I�����@a�#QH�9P?iA<��?�Unknown
�HostSlice"Yloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1(1bX9��@9bX9��@AbX9��@IbX9��@a��ϛ/P?i�$�ʞ�?�Unknown
�HostTile"Ugradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum_grad/Tile(1-����@9-����@A-����@I-����@a^�H��-P?i�'u���?�Unknown
X HostAddV2"add_9(1NbX94@9NbX94@ANbX94@INbX94@afV���PO?i��ٵ��?�Unknown
t!HostAssignAddVariableOp"AssignAddVariableOp(1��K7��@9��K7�� @A��K7��@I��K7�� @a&f5-�N?i(v�U��?�Unknown
W"HostMul"mul_14(1����S@9����S@A����S@I����S@a ��O�L?i(u �v��?�Unknown
V#HostMul"mul_3(1-����@9-����@A-����@I-����@a%)�ƲL?i2"�ew��?�Unknown
j$HostReadVariableOp"ReadVariableOp(1��x�&1
@9��x�&1�?A��x�&1
@I��x�&1�?a�����G?ip��m��?�Unknown
X%HostSqrt"Sqrt_1(1��n��	@9��n��	@A��n��	@I��n��	@a�ǆ�pG?i"d�I��?�Unknown
p&HostAssignVariableOp"AssignVariableOp_6(1h��|?5	@9h��|?5	@Ah��|?5	@Ih��|?5	@a�W{�_�F?i��n��?�Unknown
�'HostSum"Aloss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a�4���<F?i�#����?�Unknown
w(Host_MklInputConversion"MklInputConversion/_35(1R���Q@9R���Q@AR���Q@IR���Q@aَJ-y"F?i��n1��?�Unknown
q)HostReadVariableOp"mul_13/ReadVariableOp(1+����@9+����@A+����@I+����@a!��fu�E?iݵ�N���?�Unknown
V*HostMul"mul_1(19��v��@99��v��@A9��v��@I9��v��@a�E�O�D?i��"���?�Unknown
W+HostMul"mul_18(1B`��"�@9B`��"�@AB`��"�@IB`��"�@a�I��^�C?i��f:���?�Unknown
�,HostAssignAddVariableOp"0metrics/categorical_accuracy/AssignAddVariableOp(1ףp=
�@9ףp=
�@Aףp=
�@Iףp=
�@a:{%f��C?i8�c���?�Unknown
\-Host	_MklAddV2"add_1(1�p=
ף@9�p=
ף@A�p=
ף@I�p=
ף@a e?�
�C?i�?���?�Unknown
w.Host_MklInputConversion"MklInputConversion/_37(1�� �rh@9�� �rh@A�� �rh@I�� �rh@a��g��{C?i��u�s��?�Unknown
Z/Host_MklMul"mul_2(1�v��/@9�v��/@A�v��/@I�v��/@a�e���GC?i�p�E�?�Unknown
�0Host_MklReshape"[loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1(1�|?5^�@9�|?5^�@A�|?5^�@I�|?5^�@afY֌�B?iG��;��?�Unknown
g1HostCast"metrics/accuracy/Cast(1P��n�@9P��n�@AP��n�@IP��n�@a|�9�A?iLJ$�m�?�Unknown
n2HostAssignVariableOp"AssignVariableOp(1�rh��|@9�rh��|@A�rh��|@I�rh��|@a<��r��A?i�����?�Unknown
l3HostMinimum"clip_by_value_1/Minimum(1#��~j�@9#��~j�@A#��~j�@I#��~j�@a��kA?i?��M �?�Unknown
w4Host_MklInputConversion"MklInputConversion/_32(1����K@9����K@A����K@I����K@a)���@?i�I~J�?�Unknown
i5HostEqual"metrics/accuracy/Equal(1J+�@9J+�@AJ+�@IJ+�@a�@|�pv@?i�蟣g�?�Unknown
e6HostSum"metrics/accuracy/Sum(1J+�@9J+�@AJ+�@IJ+�@a�@|�pv@?i��?�"�?�Unknown
T7HostPow"Pow(1�x�&1@9�x�&1@A�x�&1@I�x�&1@a���f� ??i�]Ye&�?�Unknown
�8HostSum"Vgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1(1��n�� @9��n�� @A��n�� @I��n�� @a��)P~>?ic#5*�?�Unknown
V9HostSub"sub_6(1��K7� @9��K7� @A��K7� @I��K7� @a�k}�>?i�X�-�?�Unknown
V:HostSub"sub_4(1���Mb @9���Mb @A���Mb @I���Mb @aY���==?i���1�?�Unknown
�;Host_MklReshape"Xgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�ώ-��;?i�4Y`5�?�Unknown
Z<Host_MklMul"mul_5(1+�����?9+�����?A+�����?I+�����?a�h��oF;?i��V.8�?�Unknown
X=HostAddV2"add_5(1�Q����?9�Q����?A�Q����?I�Q����?a7�t�@;;?ix�r��;�?�Unknown
l>HostMinimum"clip_by_value_2/Minimum(1�Q����?9�Q����?A�Q����?I�Q����?a7�t�@;;?i���M?�?�Unknown
q?HostReadVariableOp"mul_18/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a6XA@�L:?iC�֓�B�?�Unknown
w@Host_MklInputConversion"MklInputConversion/_36(1�������?9�������?A�������?I�������?a��l!L6:?i��Z]�E�?�Unknown
mAHostReadVariableOp"ReadVariableOp_13(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���3:?i����!I�?�Unknown
wBHost_MklInputConversion"MklInputConversion/_34(1-�����?9-�����?A-�����?I-�����?a��8��G9?i�.��JL�?�Unknown
ZCHostSquare"Square(1ffffff�?9ffffff�?Affffff�?Iffffff�?av�8?i8�KO�?�Unknown
wDHost_MklInputConversion"MklInputConversion/_33(1+����?9+����?A+����?I+����?a��-,}7?i�hec;R�?�Unknown
XEHostAddV2"add_3(1�������?9�������?A�������?I�������?as�L7?iO��$U�?�Unknown
XFHostAddV2"add_2(1/�$��?9/�$��?A/�$��?I/�$��?a�E6TW67?iQ��X�?�Unknown
VGHostMul"mul_6(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a�3~�7?i�����Z�?�Unknown
pHHostAssignVariableOp"AssignVariableOp_7(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a�*"l�6?i#�x��]�?�Unknown
dIHostMaximum"clip_by_value_1(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a�*"l�6?ih����`�?�Unknown
VJHostMul"mul_8(1^�I+�?9^�I+�?A^�I+�?I^�I+�?a~SM��6?i����c�?�Unknown
^KHostRealDiv"	truediv_1(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?ai��>m6?ihj�$Xf�?�Unknown
pLHostAssignVariableOp"AssignVariableOp_1(1��/�$�?9��/�$�?A��/�$�?I��/�$�?a|�Jw�5?i���Si�?�Unknown
pMHostAssignVariableOp"AssignVariableOp_2(1j�t��?9j�t��?Aj�t��?Ij�t��?a�9Sv��5?ix6��k�?�Unknown
VNHostCast"Cast(17�A`���?97�A`���?A7�A`���?I7�A`���?a����4?i��Ikn�?�Unknown
�OHostCast"Oloss/activation_2_loss/categorical_crossentropy/weighted_loss/num_elements/Cast(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?aCvj��A4?i+��{�p�?�Unknown
VPHostMul"mul_4(1���K7�?9���K7�?A���K7�?I���K7�?a���@O3?i��c]s�?�Unknown
\QHostSquare"Square_1(1�/�$�?9�/�$�?A�/�$�?I�/�$�?a�S`�"3?iu����u�?�Unknown
mRHostRealDiv"metrics/accuracy/truediv(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?ap�/�1?i9�{z�w�?�Unknown
VSHostSub"sub_5(1����K�?9����K�?A����K�?I����K�?aL-5�1?i�aBw/z�?�Unknown
VTHostSub"sub_8(1�v��/�?9�v��/�?A�v��/�?I�v��/�?a@�ϻ�u1?i���0^|�?�Unknown
jUHostMinimum"clip_by_value/Minimum(1%��C��?9%��C��?A%��C��?I%��C��?a@�0?i���Fz~�?�Unknown
�VHost
Reciprocal"\gradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv(1P��n��?9P��n��?AP��n��?IP��n��?a����:�0?ilyIn���?�Unknown
WWHostMul"mul_10(1�&1��?9�&1��?A�&1��?I�&1��?a�v1|0?i�?K車�?�Unknown
�XHost_MklReshape"rgradients/loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape(1��(\���?9��(\���?A��(\���?I��(\���?a˵�G*0?i�9�/���?�Unknown
VYHostAddV2"add(1���S��?9���S��?A���S��?I���S��?a����0?i�	N-���?�Unknown
`ZHost_MklToTf"
Mkl2Tf/_30(1� �rh��?9� �rh��?A� �rh��?I� �rh��?a�L��/?i�NO֫��?�Unknown
T[HostMul"mul(1�rh��|�?9�rh��|�?A�rh��|�?I�rh��|�?a����H�/?i��*���?�Unknown
p\HostAssignVariableOp"AssignVariableOp_8(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a�u/.�/?i,��M���?�Unknown
�]HostMul"Aloss/activation_2_loss/categorical_crossentropy/weighted_loss/mul(1J+��?9J+��?AJ+��?IJ+��?a����/?i�������?�Unknown
W^HostMul"mul_17(1��"��~�?9��"��~�?A��"��~�?I��"��~�?a9��.?i�4lt��?�Unknown
p_HostAssignVariableOp"AssignVariableOp_5(1X9��v�?9X9��v�?AX9��v�?IX9��v�?a�,O�-?i�|��S��?�Unknown
``Host_MklToTf"
Mkl2Tf/_31(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�wI�1�-?iG��2��?�Unknown
iaHostCast"metrics/accuracy/Cast_1(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?ah� Ѐ-?iu��
��?�Unknown
WbHostSub"sub_11(1-�����?9-�����?A-�����?I-�����?aI�2ݱ�,?i��%Iٗ�?�Unknown
qcHostAssignVariableOp"AssignVariableOp_10(1���Q��?9���Q��?A���Q��?I���Q��?aV���+?itF�����?�Unknown
qdHostAssignVariableOp"AssignVariableOp_11(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���1�+?i0���W��?�Unknown
WeHostMul"mul_20(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���1�+?i���i��?�Unknown
yfHostRealDiv"$metrics/categorical_accuracy/truediv(1-����?9-����?A-����?I-����?a�* ��+?i���Ԟ�?�Unknown
�gHostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?a�֤D�+?i�9婍��?�Unknown
bhHostMaximum"clip_by_value(1��/�$�?9��/�$�?A��/�$�?I��/�$�?aNI�qo+?i�J�D��?�Unknown
piHostAssignVariableOp"AssignVariableOp_3(1�n����?9�n����?A�n����?I�n����?a���
n+?i���w���?�Unknown
WjHostSub"sub_12(1��MbX�?9��MbX�?A��MbX�?I��MbX�?af�w%�*?igM�ȡ��?�Unknown
okHostReadVariableOp"Cast/ReadVariableOp(1�"��~j�?9�"��~j�?A�"��~j�?I�"��~j�?a�"���)?i��ߕ?��?�Unknown
VlHostSub"sub_2(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?a�V)?i����Ԩ�?�Unknown
pmHostAssignVariableOp"AssignVariableOp_9(1+����?9+����?A+����?I+����?a�x*8*O)?ibr�i��?�Unknown
lnHostMinimum"clip_by_value_4/Minimum(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a�>G�@@)?i��~����?�Unknown
�oHostAssignAddVariableOp"2metrics/categorical_accuracy/AssignAddVariableOp_1(1�G�z��?9�G�z��?A�G�z��?I�G�z��?aDddW1)?i.����?�Unknown
`pHost_MklToTf"
Mkl2Tf/_22(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?a,��q&�(?i�J\� ��?�Unknown
WqHostMul"mul_16(1����S�?9����S�?A����S�?I����S�?a�C�S�(?i�*�Ӯ��?�Unknown
\rHostRealDiv"truediv(1
ףp=
�?9
ףp=
�?A
ףp=
�?I
ףp=
�?a#��A9�(?iC*�8��?�Unknown
�sHostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1d;�O���?9d;�O���?Ad;�O���?Id;�O���?a�$M\�3(?i��Ի��?�Unknown
VtHostSub"sub_3(1X9��v�?9X9��v�?AX9��v�?IX9��v�?aS���(?i?��4=��?�Unknown
VuHostPow"Pow_1(1=
ףp=�?9=
ףp=�?A=
ףp=�?I=
ףp=�?a:d���'?i��!R���?�Unknown
qvHostReadVariableOp"mul_11/ReadVariableOp(1���S��?9���S��?A���S��?I���S��?a����Ϗ'?i��O4��?�Unknown
pwHostReadVariableOp"mul_1/ReadVariableOp(1�n����?9�n����?A�n����?I�n����?a̴�qy'?i��9櫹�?�Unknown
VxHostSub"sub_9(1�������?9�������?A�������?I�������?as�L'?i�#�� ��?�Unknown
`yHost_MklToTf"
Mkl2Tf/_20(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a�3~�'?iqۍ��?�Unknown
yzHostReadVariableOp"dense_1/MatMul/ReadVariableOp(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?ab\����&?i��m��?�Unknown
`{Host_MklToTf"
Mkl2Tf/_21(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a��G歨&?i�
��k��?�Unknown
V|HostSub"sub_7(1�������?9�������?A�������?I�������?a2�r�O�&?i������?�Unknown
p}HostReadVariableOp"mul_8/ReadVariableOp(1��C�l�?9��C�l�?A��C�l�?I��C�l�?ax^�b�Q%?i���8*��?�Unknown
p~HostReadVariableOp"mul_3/ReadVariableOp(11�Zd�?91�Zd�?A1�Zd�?I1�Zd�?aP��@J%?i?���~��?�Unknown
yHostReadVariableOp"dense_2/MatMul/ReadVariableOp(1����S�?9����S�?A����S�?I����S�?a �DW;%?i��X����?�Unknown
]�HostSquare"Square_3(1��v���?9��v���?A��v���?I��v���?a�:mQ&%?i��#��?�Unknown
r�HostReadVariableOp"mul_16/ReadVariableOp(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a�?(!9�$?i�Ohm��?�Unknown
W�HostSqrt"Sqrt(1�t�V�?9�t�V�?A�t�V�?I�t�V�?a'�Z5T$?i�������?�Unknown
m�HostReadVariableOp"ReadVariableOp_5(1j�t��?9j�t��?Aj�t��?Ij�t��?a�[VI�	$?i�9
F���?�Unknown
{�HostReadVariableOp"dense_2/BiasAdd/ReadVariableOp(1�n����?9�n����?A�n����?I�n����?a�Vu�#?i��_�0��?�Unknown
W�HostMul"mul_9(1��K7��?9��K7��?A��K7��?I��K7��?a>&.�ϙ#?i��Z:j��?�Unknown
Y�HostSqrt"Sqrt_2(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?avv&�t#?i��܂���?�Unknown
Z�HostAddV2"add_10(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?a֟�R�V#?i�*2����?�Unknown
X�HostSub"sub_13(1���K7�?9���K7�?A���K7�?I���K7�?a���@O#?i�<���?�Unknown
��HostMul"Vgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1(1J+��?9J+��?AJ+��?IJ+��?a���m1#?i4��>��?�Unknown
Y�HostAddV2"add_4(1�/�$�?9�/�$�?A�/�$�?I�/�$�?a�S`�"#?iy�^!q��?�Unknown
m�HostReadVariableOp"ReadVariableOp_2(1!�rh���?9!�rh���?A!�rh���?I!�rh���?aF|?A&#?iq�����?�Unknown
��HostReadVariableOp"3metrics/categorical_accuracy/truediv/ReadVariableOp(1%��C��?9%��C��?A%��C��?I%��C��?ae�ŭ�"?i3����?�Unknown
n�HostReadVariableOp"ReadVariableOp_17(1�"��~j�?9�"��~j�?A�"��~j�?I�"��~j�?aũ%�ڔ"?i�<N\���?�Unknown
{�HostReadVariableOp"dense_1/BiasAdd/ReadVariableOp(1����Mb�?9����Mb�?A����Mb�?I����Mb�?a�4=f�"?i��2��?�Unknown
W�HostMul"mul_7(1'1�Z�?9'1�Z�?A'1�Z�?I'1�Z�?auoB��"?i��ʑG��?�Unknown
m�HostReadVariableOp"ReadVariableOp_7(1R���Q�?9R���Q�?AR���Q�?IR���Q�?aM�P�|~"?iɗyo��?�Unknown
n�HostReadVariableOp"ReadVariableOp_12(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a��mi�o"?i�_�r���?�Unknown
_�HostRealDiv"	truediv_2(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a��mi�o"?i��l���?�Unknown
e�HostMaximum"clip_by_value_2(1�MbX9�?9�MbX9�?A�MbX9�?I�MbX9�?a��{�h"?iu>�����?�Unknown
��HostReadVariableOp"'metrics/accuracy/truediv/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a���J5Y"?i�D�	��?�Unknown
U�HostSub"sub(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a�K�vb;"?i&Ul7-��?�Unknown
n�HostReadVariableOp"ReadVariableOp_10(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?aT(be��!?iI��DL��?�Unknown
��HostReadVariableOp"+metrics/categorical_accuracy/ReadVariableOp(1V-���?9V-���?AV-���?IV-���?a�P�Fu�!?i��i��?�Unknown
q�HostReadVariableOp"mul_6/ReadVariableOp(1����K�?9����K�?A����K�?I����K�?aL-5�!?i�eZ���?�Unknown
��HostRealDiv"Eloss/activation_2_loss/categorical_crossentropy/weighted_loss/truediv(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a$�+�q�!?i�hrq���?�Unknown
m�HostReadVariableOp"ReadVariableOp_1(1^�I+�?9^�I+�?A^�I+�?I^�I+�?a��Var!?i~�����?�Unknown
Y�HostAddV2"add_6(1`��"���?9`��"���?A`��"���?I`��"���?a�	�#WE!?i�����?�Unknown
a�Host_MklToTf"
Mkl2Tf/_25(1�l�����?9�l�����?A�l�����?I�l�����?a�l�n�=!?i��A����?�Unknown
W�HostSub"sub_1(1�l�����?9�l�����?A�l�����?I�l�����?a�l�n�=!?iU�h����?�Unknown
n�HostReadVariableOp"ReadVariableOp_11(1���Q��?9���Q��?A���Q��?I���Q��?a{  |�	!?iWR�?���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_29(1�I+��?9�I+��?A�I+��?I�I+��?a�qv>�� ?i�9���?�Unknown
m�HostReadVariableOp"ReadVariableOp_6(1��"��~�?9��"��~�?A��"��~�?I��"��~�?acԄ��� ?i��f��?�Unknown
Z�HostAddV2"add_12(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���j"� ?i}Y&��?�Unknown
e�HostMaximum"clip_by_value_4(1�t�V�?9�t�V�?A�t�V�?I�t�V�?a��� 9� ?i׉�\1��?�Unknown
Z�HostAddV2"add_11(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?aJ��O� ?ip��q;��?�Unknown
o�HostReadVariableOp"Pow/ReadVariableOp(1�G�z�?9�G�z�?A�G�z�?I�G�z�?aZ�?Y�t ?in�ºB��?�Unknown
n�HostReadVariableOp"ReadVariableOp_15(1�C�l���?9�C�l���?A�C�l���?I�C�l���?a�k:5^ ?i3�H��?�Unknown
_�HostRealDiv"	truediv_4(1V-��?9V-��?AV-��?IV-��?a{{�� ?i��OJ��?�Unknown
Y�HostSqrt"Sqrt_4(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a*A	t1 ?izR�K��?�Unknown
m�HostReadVariableOp"ReadVariableOp_8(1u�V�?9u�V�?Au�V�?Iu�V�?a�hv�?i=�rC��?�Unknown
n�HostReadVariableOp"ReadVariableOp_14(1#��~j��?9#��~j��?A#��~j��?I#��~j��?a�!8S�v?i���)7��?�Unknown
n�HostReadVariableOp"ReadVariableOp_16(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a�r�J?irIzz)��?�Unknown
n�HostReadVariableOp"ReadVariableOp_19(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?aR8��5;?i̦'T��?�Unknown
n�HostReadVariableOp"ReadVariableOp_18(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a�>�qj?i������?�Unknown
|�HostReadVariableOp"metrics/accuracy/ReadVariableOp(1���Mb�?9���Mb�?A���Mb�?I���Mb�?aY���=?i_�c����?�Unknown
n�HostReadVariableOp"ReadVariableOp_20(1V-����?9V-����?AV-����?IV-����?a��e�?i��.���?�Unknown
q�HostReadVariableOp"Pow_1/ReadVariableOp(1+���?9+���?A+���?I+���?a���]y?iR������?�Unknown
m�HostReadVariableOp"ReadVariableOp_4(1�I+��?9�I+��?A�I+��?I�I+��?a/�cL��?ioNL ���?�Unknown
m�HostReadVariableOp"ReadVariableOp_3(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?a�֤D�?i%uqjt��?�Unknown
X�HostMul"mul_19(1�v��/�?9�v��/�?A�v��/�?I�v��/�?a�޿�ď?i$[��H��?�Unknown
m�HostReadVariableOp"ReadVariableOp_9(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?a�V?ič���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_24(15^�I�?95^�I�?A5^�I�?I5^�I�?a���Ĕ?i�(�C���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_28(1�l�����?9�l�����?A�l�����?I�l�����?a��"ۅ?i>�r���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_23(1�&1��?9�&1��?A�&1��?I�&1��?a��$� �?i4O��Z��?�Unknown
a�Host_MklToTf"
Mkl2Tf/_27(1���Q��?9���Q��?A���Q��?I���Q��?a�֭�?i     �?�Unknown*��
mHost_MklSoftmax"activation_2/Softmax(1�K7�A"�@9�K7�A"�@A�K7�A"�@I�K7�A"�@aԖRo���?iԖRo���?�Unknown
nHost_MklFusedMatMul"activation_1/Relu(1F���Զ�@9F���Զ�@AF���Զ�@IF���Զ�@a��D:��?î�Hh��?�Unknown
lHost_MklFusedMatMul"dense_2/BiasAdd(1NbX9Br@9NbX9Br@ANbX9Br@INbX9Br@ah�u_e��?iy3��h�?�Unknown
�HostSoftmaxCrossEntropyWithLogits"Qloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits(1V-��q@9V-��q@AV-��q@IV-��q@aCC�XPF�?i�{�����?�Unknown
|Host
_MklMatMul"$gradients/dense_2/MatMul_grad/MatMul(1���Qn@9���Qn@A���Qn@I���Qn@aI�<�E�?i��1Ǘ�?�Unknown
�Host_MklReluGrad")gradients/activation_1/Relu_grad/ReluGrad(1-����g@9-����g@A-����g@I-����g@aj�x�G�?iA�����?�Unknown
~Host
_MklMatMul"&gradients/dense_2/MatMul_grad/MatMul_1(1�V�\@9�V�\@A�V�\@I�V�\@a60Ƞ���?iD�x���?�Unknown
kHostArgMax"metrics/accuracy/ArgMax(1F�����A@9F�����A@AF�����A@IF�����A@a{N!�j�?i~����)�?�Unknown
m	HostArgMax"metrics/accuracy/ArgMax_1(1�A`�Ђ>@9�A`�Ђ>@A�A`�Ђ>@I�A`�Ђ>@a9�
e傁?i���E�o�?�Unknown
�
HostBiasAddGrad"*gradients/dense_1/BiasAdd_grad/BiasAddGrad(1D�l���=@9D�l���=@AD�l���=@ID�l���=@a�|x���?i¢����?�Unknown
~Host
_MklMatMul"&gradients/dense_1/MatMul_grad/MatMul_1(1�E����:@9�E����:@A�E����:@I�E����:@a���M��~?i��Zxa��?�Unknown
WHostMul"mul_13(1���S��2@9���S��2@A���S��2@I���S��2@a� )�B�u?i�����?�Unknown
�HostMul"dgradients/loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits_grad/mul(1�C�l��0@9�C�l��0@A�C�l��0@I�C�l��0@aqə��~s?i�(z��C�?�Unknown
XHostSqrt"Sqrt_3(1�&1�*@9�&1�*@A�&1�*@I�&1�*@a��Uf �m?iK~�úa�?�Unknown
�HostBiasAddGrad"*gradients/dense_2/BiasAdd_grad/BiasAddGrad(1      &@9      &@A      &@I      &@a���@i?i@u���z�?�Unknown
fHostGreaterEqual"GreaterEqual(15^�I�!@95^�I�!@A5^�I�!@I5^�I�!@a,��IDbd?i�q?�]��?�Unknown
[Host_MklMul"mul_12(1�Zd; @9�Zd; @A�Zd; @I�Zd; @a�)Lءb?i�����?�Unknown
\Host	_MklAddV2"add_7(1}?5^�	 @9}?5^�	 @A}?5^�	 @I}?5^�	 @ao��O�hb?iWgۂh��?�Unknown
pHostAssignVariableOp"AssignVariableOp_4(1�ʡE�s@9�ʡE�s@A�ʡE�s@I�ʡE�s@aG�4:za?ij,����?�Unknown
WHostMul"mul_11(1����x�@9����x�@A����x�@I����x�@a�5�[�]?i�����?�Unknown
lHostMinimum"clip_by_value_3/Minimum(1ˡE���@9ˡE���@AˡE���@IˡE���@a���'�[?i�xt����?�Unknown
WHostSub"sub_10(1�I+�@9�I+�@A�I+�@I�I+�@a��߆�[?i�����?�Unknown
XHostAddV2"add_8(1F����x@9F����x@AF����x@IF����x@a��u�Y�Z?i�#�����?�Unknown
[Host_MklMul"mul_15(1Zd;�O@9Zd;�O@AZd;�O@IZd;�O@a�*5�PY?i���(
�?�Unknown
dHostMaximum"clip_by_value_3(1����K@9����K@A����K@I����K@a�	�v�qX?i�
͒a�?�Unknown
\HostSquare"Square_2(1H�z�G@9H�z�G@AH�z�G@IH�z�G@a�[��8mX?i��7/�"�?�Unknown
^HostRealDiv"	truediv_3(1
ףp=
@9
ףp=
@A
ףp=
@I
ףp=
@a2�k�� W?i�4�.�?�Unknown
�HostConcatV2"Zloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/concat_1(1�����@9�����@A�����@I�����@a��)�vT?ix�$T8�?�Unknown
�HostSlice"Yloss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Slice_1(1bX9��@9bX9��@AbX9��@IbX9��@a
�/�iT?i]C<��B�?�Unknown
�HostTile"Ugradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum_grad/Tile(1-����@9-����@A-����@I-����@ar�^�gT?i�k��L�?�Unknown
XHostAddV2"add_9(1NbX94@9NbX94@ANbX94@INbX94@a�"�x�S?i���}�V�?�Unknown
t HostAssignAddVariableOp"AssignAddVariableOp(1��K7��@9��K7�� @A��K7��@I��K7�� @a���;S?i�Pg:`�?�Unknown
W!HostMul"mul_14(1����S@9����S@A����S@I����S@a������Q?i���7i�?�Unknown
V"HostMul"mul_3(1-����@9-����@A-����@I-����@a����ԩQ?i�ȉ�r�?�Unknown
j#HostReadVariableOp"ReadVariableOp(1��x�&1
@9��x�&1�?A��x�&1
@I��x�&1�?a�ɉ�N?i�:,�y�?�Unknown
X$HostSqrt"Sqrt_1(1��n��	@9��n��	@A��n��	@I��n��	@a�h��V�M?ig�����?�Unknown
p%HostAssignVariableOp"AssignVariableOp_6(1h��|?5	@9h��|?5	@Ah��|?5	@Ih��|?5	@atJWo{�L?i:Ĺ�0��?�Unknown
�&HostSum"Aloss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum(1;�O��n@9;�O��n@A;�O��n@I;�O��n@a�O�}sL?i�;��3��?�Unknown
w'Host_MklInputConversion"MklInputConversion/_35(1R���Q@9R���Q@AR���Q@IR���Q@au�	��K?i1~&.��?�Unknown
q(HostReadVariableOp"mul_13/ReadVariableOp(1+����@9+����@A+����@I+����@a���I�K?i������?�Unknown
V)HostMul"mul_1(19��v��@99��v��@A9��v��@I9��v��@a�y�)�I?i5�����?�Unknown
W*HostMul"mul_18(1B`��"�@9B`��"�@AB`��"�@IB`��"�@a_��n{I?ik��ҩ�?�Unknown
�+HostAssignAddVariableOp"0metrics/categorical_accuracy/AssignAddVariableOp(1ףp=
�@9ףp=
�@Aףp=
�@Iףp=
�@aL(��I?iu!���?�Unknown
\,Host	_MklAddV2"add_1(1�p=
ף@9�p=
ף@A�p=
ף@I�p=
ף@ae�ou�H?i_}��L��?�Unknown
w-Host_MklInputConversion"MklInputConversion/_37(1�� �rh@9�� �rh@A�� �rh@I�� �rh@aZ�m�ՒH?i�X�}q��?�Unknown
Z.Host_MklMul"mul_2(1�v��/@9�v��/@A�v��/@I�v��/@aXG�QH?i�ɽ����?�Unknown
�/Host_MklReshape"[loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_1(1�|?5^�@9�|?5^�@A�|?5^�@I�|?5^�@aK���G?iP�x��?�Unknown
g0HostCast"metrics/accuracy/Cast(1P��n�@9P��n�@AP��n�@IP��n�@a�8���eF?il�����?�Unknown
n1HostAssignVariableOp"AssignVariableOp(1�rh��|@9�rh��|@A�rh��|@I�rh��|@a�3���^F?iy�I����?�Unknown
l2HostMinimum"clip_by_value_1/Minimum(1#��~j�@9#��~j�@A#��~j�@I#��~j�@aT>,��E?i��K�	��?�Unknown
w3Host_MklInputConversion"MklInputConversion/_32(1����K@9����K@A����K@I����K@aX�1E\ E?i�7]J��?�Unknown
i4HostEqual"metrics/accuracy/Equal(1J+�@9J+�@AJ+�@IJ+�@ai�8=�D?iF��z��?�Unknown
e5HostSum"metrics/accuracy/Sum(1J+�@9J+�@AJ+�@IJ+�@ai�8=�D?iFT볫��?�Unknown
T6HostPow"Pow(1�x�&1@9�x�&1@A�x�&1@I�x�&1@a�ӫ��C?i;�V���?�Unknown
�7HostSum"Vgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/mul_grad/Sum_1(1��n�� @9��n�� @A��n�� @I��n�� @ae�2�:C?i6��]��?�Unknown
V8HostSub"sub_6(1��K7� @9��K7� @A��K7� @I��K7� @am�F9-�B?i��1d��?�Unknown
V9HostSub"sub_4(1���Mb @9���Mb @A���Mb @I���Mb @aM��4zpB?i�1�����?�Unknown
�:Host_MklReshape"Xgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/Sum_grad/Reshape(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a{�V�IwA?i<G;U �?�Unknown
Z;Host_MklMul"mul_5(1+�����?9+�����?A+�����?I+�����?ap�TV3A?ioܐc�?�Unknown
X<HostAddV2"add_5(1�Q����?9�Q����?A�Q����?I�Q����?aU�K�,A?i`/� ��?�Unknown
l=HostMinimum"clip_by_value_2/Minimum(1�Q����?9�Q����?A�Q����?I�Q����?aU�K�,A?iQ��$��?�Unknown
q>HostReadVariableOp"mul_18/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a6ϛ�@?i�Ow��?�Unknown
w?Host_MklInputConversion"MklInputConversion/_36(1�������?9�������?A�������?I�������?a��#퀇@?iИ�k@�?�Unknown
m@HostReadVariableOp"ReadVariableOp_13(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���:w@?i6G�.^�?�Unknown
wAHost_MklInputConversion"MklInputConversion/_34(1-�����?9-�����?A-�����?I-�����?am��??i�
�qZ�?�Unknown
ZBHostSquare"Square(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�����M>?ic-*$!�?�Unknown
wCHost_MklInputConversion"MklInputConversion/_33(1+����?9+����?A+����?I+����?a(�8y˟=?i��#�$�?�Unknown
XDHostAddV2"add_3(1�������?9�������?A�������?I�������?a8�?P�b=?i�&y�(�?�Unknown
XEHostAddV2"add_2(1/�$��?9/�$��?A/�$��?I/�$��?a���vF=?i��G-,�?�Unknown
VFHostMul"mul_6(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a5Er=?i7�@6�/�?�Unknown
pGHostAssignVariableOp"AssignVariableOp_7(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a���!��<?il�D4p3�?�Unknown
dHHostMaximum"clip_by_value_1(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a���!��<?i�,I27�?�Unknown
VIHostMul"mul_8(1^�I+�?9^�I+�?A^�I+�?I^�I+�?aF��]��<?iT����:�?�Unknown
^JHostRealDiv"	truediv_1(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a�&צ�H<?i9��5>�?�Unknown
pKHostAssignVariableOp"AssignVariableOp_1(1��/�$�?9��/�$�?A��/�$�?I��/�$�?a�r1Ҷ;?i{�/��A�?�Unknown
pLHostAssignVariableOp"AssignVariableOp_2(1j�t��?9j�t��?Aj�t��?Ij�t��?a��+5�;?igռE�?�Unknown
VMHostCast"Cast(17�A`���?97�A`���?A7�A`���?I7�A`���?a��)�0:?iClx�dH�?�Unknown
�NHostCast"Oloss/activation_2_loss/categorical_crossentropy/weighted_loss/num_elements/Cast(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a+� �9?i~l<P�K�?�Unknown
VOHostMul"mul_4(1���K7�?9���K7�?A���K7�?I���K7�?a}�%SjZ8?i2ц��N�?�Unknown
\PHostSquare"Square_1(1�/�$�?9�/�$�?A�/�$�?I�/�$�?a�z���!8?i�,`ݥQ�?�Unknown
mQHostRealDiv"metrics/accuracy/truediv(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a\R�.�6?i+��vT�?�Unknown
VRHostSub"sub_5(1����K�?9����K�?A����K�?I����K�?a�
T 7&6?i���;W�?�Unknown
VSHostSub"sub_8(1�v��/�?9�v��/�?A�v��/�?I�v��/�?aXH�M6?i��Yl�Y�?�Unknown
jTHostMinimum"clip_by_value/Minimum(1%��C��?9%��C��?A%��C��?I%��C��?av�<I5?i~铥\�?�Unknown
�UHost
Reciprocal"\gradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/truediv_grad/RealDiv(1P��n��?9P��n��?AP��n��?IP��n��?aQ��>�?5?ioN��M_�?�Unknown
WVHostMul"mul_10(1�&1��?9�&1��?A�&1��?I�&1��?a��A�J�4?i�����a�?�Unknown
�WHost_MklReshape"rgradients/loss/activation_2_loss/categorical_crossentropy/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape(1��(\���?9��(\���?A��(\���?I��(\���?a����b4?i��:3sd�?�Unknown
VXHostAddV2"add(1���S��?9���S��?A���S��?I���S��?an>Y�A4?i��eq�f�?�Unknown
`YHost_MklToTf"
Mkl2Tf/_30(1� �rh��?9� �rh��?A� �rh��?I� �rh��?a��5o*4?i�L��i�?�Unknown
TZHostMul"mul(1�rh��|�?9�rh��|�?A�rh��|�?I�rh��|�?a�4W�4?i���l�?�Unknown
p[HostAssignVariableOp"AssignVariableOp_8(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a!l��3?ii|0�n�?�Unknown
�\HostMul"Aloss/activation_2_loss/categorical_crossentropy/weighted_loss/mul(1J+��?9J+��?AJ+��?IJ+��?a�4ab�3?i����p�?�Unknown
W]HostMul"mul_17(1��"��~�?9��"��~�?A��"��~�?I��"��~�?a>�'l�2?i�}M�Qs�?�Unknown
p^HostAssignVariableOp"AssignVariableOp_5(1X9��v�?9X9��v�?AX9��v�?IX9��v�?a�+��2?i�Bꊮu�?�Unknown
`_Host_MklToTf"
Mkl2Tf/_31(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�Sʤ��2?iDܞ
x�?�Unknown
i`HostCast"metrics/accuracy/Cast_1(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?a�� �ʚ2?i^@�w]z�?�Unknown
WaHostSub"sub_11(1-�����?9-�����?A-�����?I-�����?a�5SM�<2?i��C�|�?�Unknown
qbHostAssignVariableOp"AssignVariableOp_10(1���Q��?9���Q��?A���Q��?I���Q��?a!Ō���1?i^��C�~�?�Unknown
qcHostAssignVariableOp"AssignVariableOp_11(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���1?i�wu���?�Unknown
WdHostMul"mul_20(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a���1?id3T}@��?�Unknown
yeHostRealDiv"$metrics/categorical_accuracy/truediv(1-����?9-����?A-����?I-����?a�h+U3�1?i�ؾ�s��?�Unknown
�fHostAssignAddVariableOp"&metrics/accuracy/AssignAddVariableOp_1(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?a@���_1?i9u�|���?�Unknown
bgHostMaximum"clip_by_value(1��/�$�?9��/�$�?A��/�$�?I��/�$�?aՇ J�L1?iJ��ɉ�?�Unknown
phHostAssignVariableOp"AssignVariableOp_3(1�n����?9�n����?A�n����?I�n����?a�bA1?il
D��?�Unknown
WiHostSub"sub_12(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a	�ߘn�0?i]&�1��?�Unknown
ojHostReadVariableOp"Cast/ReadVariableOp(1�"��~j�?9�"��~j�?A�"��~j�?I�"��~j�?a���dO0?i�����?�Unknown
VkHostSub"sub_2(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?aM%�0��/?i��c��?�Unknown
plHostAssignVariableOp"AssignVariableOp_9(1+����?9+����?A+����?I+����?a(�}��/?i����?�Unknown
lmHostMinimum"clip_by_value_4/Minimum(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a��l��/?iW�F���?�Unknown
�nHostAssignAddVariableOp"2metrics/categorical_accuracy/AssignAddVariableOp_1(1�G�z��?9�G�z��?A�G�z��?I�G�z��?a�X����/?i�2�	��?�Unknown
`oHost_MklToTf"
Mkl2Tf/_22(1��~j�t�?9��~j�t�?A��~j�t�?I��~j�t�?a��N �/?i�7�E��?�Unknown
WpHostMul"mul_16(1����S�?9����S�?A����S�?I����S�?a�b�t^/?i`��,���?�Unknown
\qHostRealDiv"truediv(1
ףp=
�?9
ףp=
�?A
ףp=
�?I
ףp=
�?a�%]��	/?i2����?�Unknown
�rHostAssignAddVariableOp"$metrics/accuracy/AssignAddVariableOp(1d;�O���?9d;�O���?Ad;�O���?Id;�O���?a�
;-�.?i�z�,П�?�Unknown
VsHostSub"sub_3(1X9��v�?9X9��v�?AX9��v�?IX9��v�?a��5�`.?i��5���?�Unknown
VtHostPow"Pow_1(1=
ףp=�?9=
ףp=�?A=
ףp=�?I=
ףp=�?a&�k�.?iѐ�!���?�Unknown
quHostReadVariableOp"mul_11/ReadVariableOp(1���S��?9���S��?A���S��?I���S��?a�0��M�-?i�[��s��?�Unknown
pvHostReadVariableOp"mul_1/ReadVariableOp(1�n����?9�n����?A�n����?I�n����?a���-?i�
HM��?�Unknown
VwHostSub"sub_9(1�������?9�������?A�������?I�������?a8�?P�b-?i��r#��?�Unknown
`xHost_MklToTf"
Mkl2Tf/_20(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a5Er-?if;�����?�Unknown
yyHostReadVariableOp"dense_1/MatMul/ReadVariableOp(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?a���<�,?ivL��Ĭ�?�Unknown
`zHost_MklToTf"
Mkl2Tf/_21(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a�̓,?i�j�ڍ��?�Unknown
V{HostSub"sub_7(1�������?9�������?A�������?I�������?a���w,?ivF�SU��?�Unknown
p|HostReadVariableOp"mul_8/ReadVariableOp(1��C�l�?9��C�l�?A��C�l�?I��C�l�?amyc�>�*?i�l���?�Unknown
p}HostReadVariableOp"mul_3/ReadVariableOp(11�Zd�?91�Zd�?A1�Zd�?I1�Zd�?aH���*?i�|`%���?�Unknown
y~HostReadVariableOp"dense_2/MatMul/ReadVariableOp(1����S�?9����S�?A����S�?I����S�?a�d?.	�*?i�`�]��?�Unknown
\HostSquare"Square_3(1��v���?9��v���?A��v���?I��v���?a�ߕd6�*?i$�Y���?�Unknown
r�HostReadVariableOp"mul_16/ReadVariableOp(1㥛� ��?9㥛� ��?A㥛� ��?I㥛� ��?a2��
*?ig�ژ���?�Unknown
W�HostSqrt"Sqrt(1�t�V�?9�t�V�?A�t�V�?I�t�V�?a�<uC��)?i�_�@��?�Unknown
m�HostReadVariableOp"ReadVariableOp_5(1j�t��?9j�t��?Aj�t��?Ij�t��?a���E)?i5fZ)ջ�?�Unknown
{�HostReadVariableOp"dense_2/BiasAdd/ReadVariableOp(1�n����?9�n����?A�n����?I�n����?a��)?i&)de��?�Unknown
W�HostMul"mul_9(1��K7��?9��K7��?A��K7��?I��K7��?a�<��r�(?iK5W���?�Unknown
Y�HostSqrt"Sqrt_2(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?a6p�n�(?i�@�y��?�Unknown
Z�HostAddV2"add_10(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?a�����c(?i�Z����?�Unknown
X�HostSub"sub_13(1���K7�?9���K7�?A���K7�?I���K7�?a}�%SjZ(?i�P�e���?�Unknown
��HostMul"Vgradients/loss/activation_2_loss/categorical_crossentropy/weighted_loss/mul_grad/Mul_1(1J+��?9J+��?AJ+��?IJ+��?a�2�M�4(?i�*Բ��?�Unknown
Y�HostAddV2"add_4(1�/�$�?9�/�$�?A�/�$�?I�/�$�?a�z���!(?i���Ҋ��?�Unknown
m�HostReadVariableOp"ReadVariableOp_2(1!�rh���?9!�rh���?A!�rh���?I!�rh���?a1f��(?iMDQ/��?�Unknown
��HostReadVariableOp"3metrics/categorical_accuracy/truediv/ReadVariableOp(1%��C��?9%��C��?A%��C��?I%��C��?aw)��'?iަp~���?�Unknown
n�HostReadVariableOp"ReadVariableOp_17(1�"��~j�?9�"��~j�?A�"��~j�?I�"��~j�?a㣣�To'?i��s���?�Unknown
{�HostReadVariableOp"dense_1/BiasAdd/ReadVariableOp(1����Mb�?9����Mb�?A����Mb�?I����Mb�?a�GB��e'?i<���q��?�Unknown
W�HostMul"mul_7(1'1�Z�?9'1�Z�?A'1�Z�?I'1�Z�?a���m�\'?iK�����?�Unknown
m�HostReadVariableOp"ReadVariableOp_7(1R���Q�?9R���Q�?AR���Q�?IR���Q�?au�,S'?iDK��\��?�Unknown
n�HostReadVariableOp"ReadVariableOp_12(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a+׼�P@'?i������?�Unknown
_�HostRealDiv"	truediv_2(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a+׼�P@'?iނ	�D��?�Unknown
e�HostMaximum"clip_by_value_2(1�MbX9�?9�MbX9�?A�MbX9�?I�MbX9�?a{[h�6'?i��E���?�Unknown
��HostReadVariableOp"'metrics/accuracy/truediv/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a��$'?i"bN�*��?�Unknown
U�HostSub"sub(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a)R�}�&?iWc,o���?�Unknown
n�HostReadVariableOp"ReadVariableOp_10(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a��ERu�&?i���v��?�Unknown
��HostReadVariableOp"+metrics/categorical_accuracy/ReadVariableOp(1V-���?9V-���?AV-���?IV-���?aI�!�?�&?i�iz�l��?�Unknown
q�HostReadVariableOp"mul_6/ReadVariableOp(1����K�?9����K�?A����K�?I����K�?a�
T 7&&?io����?�Unknown
��HostRealDiv"Eloss/activation_2_loss/categorical_crossentropy/weighted_loss/truediv(1�$��C�?9�$��C�?A�$��C�?I�$��C�?a����&?i9^��0��?�Unknown
m�HostReadVariableOp"ReadVariableOp_1(1^�I+�?9^�I+�?A^�I+�?I^�I+�?aE���� &?i#�����?�Unknown
Y�HostAddV2"add_6(1`��"���?9`��"���?A`��"���?I`��"���?ahq�r.�%?i�3mw���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_25(1�l�����?9�l�����?A�l�����?I�l�����?aC%1Ǿ%?i�E�cI��?�Unknown
W�HostSub"sub_1(1�l�����?9�l�����?A�l�����?I�l�����?aC%1Ǿ%?i,XSP���?�Unknown
n�HostReadVariableOp"ReadVariableOp_11(1���Q��?9���Q��?A���Q��?I���Q��?aA�{g�|%?i�ϙ���?�Unknown
a�Host_MklToTf"
Mkl2Tf/_29(1�I+��?9�I+��?A�I+��?I�I+��?acg3߈D%?i�'hQ��?�Unknown
m�HostReadVariableOp"ReadVariableOp_6(1��"��~�?9��"��~�?A��"��~�?I��"��~�?a>ҝ!;%?i<�A���?�Unknown
Z�HostAddV2"add_12(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�����%?i;����?�Unknown
e�HostMaximum"clip_by_value_4(1�t�V�?9�t�V�?A�t�V�?I�t�V�?a�>�V%?iϩ��G��?�Unknown
Z�HostAddV2"add_11(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?a<�(�N�$?iW��_���?�Unknown
o�HostReadVariableOp"Pow/ReadVariableOp(1�G�z�?9�G�z�?A�G�z�?I�G�z�?a_]�K��$?i]��m���?�Unknown
n�HostReadVariableOp"ReadVariableOp_15(1�C�l���?9�C�l���?A�C�l���?I�C�l���?a�H����$?i"&ϸ-��?�Unknown
_�HostRealDiv"	truediv_4(1V-��?9V-��?AV-��?IV-��?a�P;P$?i#ے�r��?�Unknown
Y�HostSqrt"Sqrt_4(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a\S��==$?i�cn����?�Unknown
m�HostReadVariableOp"ReadVariableOp_8(1u�V�?9u�V�?Au�V�?Iu�V�?a�ش��#?iF_ ����?�Unknown
n�HostReadVariableOp"ReadVariableOp_14(1#��~j��?9#��~j��?A#��~j��?I#��~j��?aS?��5#?i�}I,#��?�Unknown
n�HostReadVariableOp"ReadVariableOp_16(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a�*�ͼ#?i�Y�T��?�Unknown
n�HostReadVariableOp"ReadVariableOp_19(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?a��a�U#?i
 oͅ��?�Unknown
n�HostReadVariableOp"ReadVariableOp_18(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a�����"?i��n����?�Unknown
|�HostReadVariableOp"metrics/accuracy/ReadVariableOp(1���Mb�?9���Mb�?A���Mb�?I���Mb�?aM��4zp"?i������?�Unknown
n�HostReadVariableOp"ReadVariableOp_20(1V-����?9V-����?AV-����?IV-����?aޛ�pDT"?iY����?�Unknown
q�HostReadVariableOp"Pow_1/ReadVariableOp(1+���?9+���?A+���?I+���?ak}Oi�!?i��*��?�Unknown
m�HostReadVariableOp"ReadVariableOp_4(1�I+��?9�I+��?A�I+��?I�I+��?a��h�d�!?i��7�.��?�Unknown
m�HostReadVariableOp"ReadVariableOp_3(1��ʡE�?9��ʡE�?A��ʡE�?I��ʡE�?a@���_!?i�}D��?�Unknown
X�HostMul"mul_19(1�v��/�?9�v��/�?A�v��/�?I�v��/�?a�!lu� ?i��{|P��?�Unknown
m�HostReadVariableOp"ReadVariableOp_9(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?aM%�0��?i}s�#P��?�Unknown
a�Host_MklToTf"
Mkl2Tf/_24(15^�I�?95^�I�?A5^�I�?I5^�I�?a����k ?i[�'H��?�Unknown
a�Host_MklToTf"
Mkl2Tf/_28(1�l�����?9�l�����?A�l�����?I�l�����?aD9
��?i$-�?��?�Unknown
a�Host_MklToTf"
Mkl2Tf/_23(1�&1��?9�&1��?A�&1��?I�&1��?a��Uf �?i�_]/��?�Unknown
a�Host_MklToTf"
Mkl2Tf/_27(1���Q��?9���Q��?A���Q��?I���Q��?aA�T_?i�������?�Unknown