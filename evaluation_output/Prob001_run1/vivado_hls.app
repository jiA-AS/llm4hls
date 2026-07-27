<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="Prob001_run1" top="TopModule">
    <includePaths/>
    <libraryFlag/>
    <Simulation argv="">
        <SimFlow name="csim" setup="true" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <solutions>
        <solution name="simulation" status=""/>
        <solution name="synthesis" status=""/>
    </solutions>
    <files>
        <file name="Prob001_design_run1.cpp" sc="0" tb="false" cflags="" blackbox="false"/>
        <file name="../../Prob001_tb.cpp" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" blackbox="false"/>
    </files>
</AutoPilot:project>

