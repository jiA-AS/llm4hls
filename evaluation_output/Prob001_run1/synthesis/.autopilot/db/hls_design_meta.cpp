#include "hls_design_meta.h"
const Port_Property HLS_Design_Meta::port_props[]={
	Port_Property("zero_V", 1, hls_out, 0, "ap_none", "out_data", 1),
};
const char* HLS_Design_Meta::dut_name = "TopModule";
