#pragma once
#include "FilterBase.hpp"
#include <memory>
#include <string>
// Enum listing all available filters
enum class FilterType {
    Acopy,  // Acopy filter
    Aderivative,  // Aderivative filter
    Aintegral,  // Aintegral filter
    Alatency,  // Alatency filter
    Amultiply,  // Amultiply filter
    Anull,  // Anull filter
    Apsnr,  // Apsnr filter
    Areverse,  // Areverse filter
    Asdr,  // Asdr filter
    Ashowinfo,  // Ashowinfo filter
    Asisdr,  // Asisdr filter
    Earwax,  // Earwax filter
    Volumedetect,  // Volumedetect filter
    Anullsink,  // Anullsink filter
    Addroi,  // Addroi filter
    Alphaextract,  // Alphaextract filter
    Alphamerge,  // Alphamerge filter
    Amplify,  // Amplify filter
    Atadenoise,  // Atadenoise filter
    Avgblur,  // Avgblur filter
    Backgroundkey,  // Backgroundkey filter
    Bbox,  // Bbox filter
    Bench,  // Bench filter
    Bilateral,  // Bilateral filter
    Bitplanenoise,  // Bitplanenoise filter
    Blackdetect,  // Blackdetect filter
    Blend,  // Blend filter
    Blockdetect,  // Blockdetect filter
    Blurdetect,  // Blurdetect filter
    Bm3d,  // Bm3d filter
    Bwdif,  // Bwdif filter
    Cas,  // Cas filter
    Ccrepack,  // Ccrepack filter
    Chromahold,  // Chromahold filter
    Chromakey,  // Chromakey filter
    Chromanr,  // Chromanr filter
    Chromashift,  // Chromashift filter
    Ciescope,  // Ciescope filter
    Codecview,  // Codecview filter
    Colorbalance,  // Colorbalance filter
    Colorchannelmixer,  // Colorchannelmixer filter
    Colorcontrast,  // Colorcontrast filter
    Colorcorrect,  // Colorcorrect filter
    Colorize,  // Colorize filter
    Colorkey,  // Colorkey filter
    Colorhold,  // Colorhold filter
    Colorlevels,  // Colorlevels filter
    Colormap,  // Colormap filter
    Colorspace,  // Colorspace filter
    Colortemperature,  // Colortemperature filter
    Convolution,  // Convolution filter
    Convolve,  // Convolve filter
    Copy,  // Copy filter
    Corr,  // Corr filter
    Crop,  // Crop filter
    Curves,  // Curves filter
    Datascope,  // Datascope filter
    Dblur,  // Dblur filter
    Dctdnoiz,  // Dctdnoiz filter
    Deband,  // Deband filter
    Deblock,  // Deblock filter
    Decimate,  // Decimate filter
    Deconvolve,  // Deconvolve filter
    Dedot,  // Dedot filter
    Deflate,  // Deflate filter
    Deflicker,  // Deflicker filter
    Dejudder,  // Dejudder filter
    Derain,  // Derain filter
    Deshake,  // Deshake filter
    Despill,  // Despill filter
    Detelecine,  // Detelecine filter
    Dilation,  // Dilation filter
    Displace,  // Displace filter
    Dnn_classify,  // Dnn_classify filter
    Dnn_detect,  // Dnn_detect filter
    Dnn_processing,  // Dnn_processing filter
    Doubleweave,  // Doubleweave filter
    Drawbox,  // Drawbox filter
    Drawgraph,  // Drawgraph filter
    Drawgrid,  // Drawgrid filter
    Edgedetect,  // Edgedetect filter
    Elbg,  // Elbg filter
    Entropy,  // Entropy filter
    Epx,  // Epx filter
    Erosion,  // Erosion filter
    Estdif,  // Estdif filter
    Exposure,  // Exposure filter
    Extractplanes,  // Extractplanes filter
    Fade,  // Fade filter
    Feedback,  // Feedback filter
    Fftdnoiz,  // Fftdnoiz filter
    Fftfilt,  // Fftfilt filter
    Field,  // Field filter
    Fieldhint,  // Fieldhint filter
    Fieldmatch,  // Fieldmatch filter
    Fieldorder,  // Fieldorder filter
    Fillborders,  // Fillborders filter
    Floodfill,  // Floodfill filter
    Format,  // Format filter
    Fps,  // Fps filter
    Framepack,  // Framepack filter
    Framerate,  // Framerate filter
    Framestep,  // Framestep filter
    Freezedetect,  // Freezedetect filter
    Freezeframes,  // Freezeframes filter
    Fsync,  // Fsync filter
    Gblur,  // Gblur filter
    Geq,  // Geq filter
    Gradfun,  // Gradfun filter
    Graphmonitor,  // Graphmonitor filter
    Grayworld,  // Grayworld filter
    Greyedge,  // Greyedge filter
    Guided,  // Guided filter
    Haldclut,  // Haldclut filter
    Hflip,  // Hflip filter
    Histogram,  // Histogram filter
    Hqx,  // Hqx filter
    Hstack,  // Hstack filter
    Hsvhold,  // Hsvhold filter
    Hsvkey,  // Hsvkey filter
    Hue,  // Hue filter
    Huesaturation,  // Huesaturation filter
    Hwdownload,  // Hwdownload filter
    Hwmap,  // Hwmap filter
    Hwupload,  // Hwupload filter
    Hwupload_cuda,  // Hwupload_cuda filter
    Hysteresis,  // Hysteresis filter
    Identity,  // Identity filter
    Idet,  // Idet filter
    Il,  // Il filter
    Inflate,  // Inflate filter
    Interleave,  // Interleave filter
    Kirsch,  // Kirsch filter
    Lagfun,  // Lagfun filter
    Latency,  // Latency filter
    Lenscorrection,  // Lenscorrection filter
    Limitdiff,  // Limitdiff filter
    Limiter,  // Limiter filter
    Loop,  // Loop filter
    Lumakey,  // Lumakey filter
    Lut,  // Lut filter
    Lut1d,  // Lut1d filter
    Lut2,  // Lut2 filter
    Lut3d,  // Lut3d filter
    Lutrgb,  // Lutrgb filter
    Lutyuv,  // Lutyuv filter
    Maskedclamp,  // Maskedclamp filter
    Maskedmax,  // Maskedmax filter
    Maskedmerge,  // Maskedmerge filter
    Maskedmin,  // Maskedmin filter
    Maskedthreshold,  // Maskedthreshold filter
    Maskfun,  // Maskfun filter
    Median,  // Median filter
    Mergeplanes,  // Mergeplanes filter
    Mestimate,  // Mestimate filter
    Metadata,  // Metadata filter
    Midequalizer,  // Midequalizer filter
    Minterpolate,  // Minterpolate filter
    Mix,  // Mix filter
    Monochrome,  // Monochrome filter
    Morpho,  // Morpho filter
    Msad,  // Msad filter
    Multiply,  // Multiply filter
    Negate,  // Negate filter
    Nlmeans,  // Nlmeans filter
    Noformat,  // Noformat filter
    Noise,  // Noise filter
    Normalize,  // Normalize filter
    Null,  // Null filter
    Oscilloscope,  // Oscilloscope filter
    Overlay,  // Overlay filter
    Pad,  // Pad filter
    Palettegen,  // Palettegen filter
    Paletteuse,  // Paletteuse filter
    Photosensitivity,  // Photosensitivity filter
    Pixdesctest,  // Pixdesctest filter
    Pixelize,  // Pixelize filter
    Pixscope,  // Pixscope filter
    Premultiply,  // Premultiply filter
    Prewitt,  // Prewitt filter
    Pseudocolor,  // Pseudocolor filter
    Psnr,  // Psnr filter
    Qp,  // Qp filter
    Random,  // Random filter
    Readeia608,  // Readeia608 filter
    Readvitc,  // Readvitc filter
    Remap,  // Remap filter
    Removegrain,  // Removegrain filter
    Removelogo,  // Removelogo filter
    Reverse,  // Reverse filter
    Rgbashift,  // Rgbashift filter
    Roberts,  // Roberts filter
    Rotate,  // Rotate filter
    Scale,  // Scale filter
    Scale2ref,  // Scale2ref filter
    Scdet,  // Scdet filter
    Scharr,  // Scharr filter
    Scroll,  // Scroll filter
    Segment,  // Segment filter
    Select,  // Select filter
    Selectivecolor,  // Selectivecolor filter
    Separatefields,  // Separatefields filter
    Setdar,  // Setdar filter
    Setfield,  // Setfield filter
    Setparams,  // Setparams filter
    Setpts,  // Setpts filter
    Setrange,  // Setrange filter
    Setsar,  // Setsar filter
    Settb,  // Settb filter
    Shear,  // Shear filter
    Showinfo,  // Showinfo filter
    Showpalette,  // Showpalette filter
    Shuffleframes,  // Shuffleframes filter
    Shufflepixels,  // Shufflepixels filter
    Shuffleplanes,  // Shuffleplanes filter
    Sidedata,  // Sidedata filter
    Signalstats,  // Signalstats filter
    Siti,  // Siti filter
    Sobel,  // Sobel filter
    Sr,  // Sr filter
    Ssim,  // Ssim filter
    Ssim360,  // Ssim360 filter
    Swaprect,  // Swaprect filter
    Swapuv,  // Swapuv filter
    Tblend,  // Tblend filter
    Telecine,  // Telecine filter
    Thistogram,  // Thistogram filter
    Threshold,  // Threshold filter
    Thumbnail,  // Thumbnail filter
    Tile,  // Tile filter
    Tiltandshift,  // Tiltandshift filter
    Tlut2,  // Tlut2 filter
    Tmedian,  // Tmedian filter
    Tmidequalizer,  // Tmidequalizer filter
    Tmix,  // Tmix filter
    Tonemap,  // Tonemap filter
    Tpad,  // Tpad filter
    Transpose,  // Transpose filter
    Trim,  // Trim filter
    Unpremultiply,  // Unpremultiply filter
    Unsharp,  // Unsharp filter
    Untile,  // Untile filter
    V360,  // V360 filter
    Varblur,  // Varblur filter
    Vectorscope,  // Vectorscope filter
    Vflip,  // Vflip filter
    Vfrdet,  // Vfrdet filter
    Vibrance,  // Vibrance filter
    Vif,  // Vif filter
    Vignette,  // Vignette filter
    Vmafmotion,  // Vmafmotion filter
    Vstack,  // Vstack filter
    W3fdif,  // W3fdif filter
    Waveform,  // Waveform filter
    Weave,  // Weave filter
    Xbr,  // Xbr filter
    Xcorrelate,  // Xcorrelate filter
    Xfade,  // Xfade filter
    Xmedian,  // Xmedian filter
    Xstack,  // Xstack filter
    Yadif,  // Yadif filter
    Yaepblur,  // Yaepblur filter
    Zoompan,  // Zoompan filter
    Allrgb,  // Allrgb filter
    Allyuv,  // Allyuv filter
    Cellauto,  // Cellauto filter
    Color,  // Color filter
    Colorchart,  // Colorchart filter
    Colorspectrum,  // Colorspectrum filter
    Ddagrab,  // Ddagrab filter
    Gradients,  // Gradients filter
    Haldclutsrc,  // Haldclutsrc filter
    Life,  // Life filter
    Mandelbrot,  // Mandelbrot filter
    Nullsrc,  // Nullsrc filter
    Pal75bars,  // Pal75bars filter
    Pal100bars,  // Pal100bars filter
    Rgbtestsrc,  // Rgbtestsrc filter
    Sierpinski,  // Sierpinski filter
    Smptebars,  // Smptebars filter
    Smptehdbars,  // Smptehdbars filter
    Testsrc,  // Testsrc filter
    Testsrc2,  // Testsrc2 filter
    Yuvtestsrc,  // Yuvtestsrc filter
    Zoneplate,  // Zoneplate filter
    Nullsink,  // Nullsink filter
    A3dscope,  // A3dscope filter
    Abitscope,  // Abitscope filter
    Adrawgraph,  // Adrawgraph filter
    Agraphmonitor,  // Agraphmonitor filter
    Ahistogram,  // Ahistogram filter
    Aphasemeter,  // Aphasemeter filter
    Avectorscope,  // Avectorscope filter
    Showcqt,  // Showcqt filter
    Showcwt,  // Showcwt filter
    Showfreqs,  // Showfreqs filter
    Showspatial,  // Showspatial filter
    Showspectrum,  // Showspectrum filter
    Showspectrumpic,  // Showspectrumpic filter
    Showvolume,  // Showvolume filter
    Showwaves,  // Showwaves filter
    Showwavespic,  // Showwavespic filter
    Buffer,  // Buffer filter
    Buffersink,  // Buffersink filter
};

// Factory function to create filters
std::shared_ptr<FilterBase> CreateFilter(FilterType type);


#include "Acopy.hpp"
#include "Aderivative.hpp"
#include "Aintegral.hpp"
#include "Alatency.hpp"
#include "Amultiply.hpp"
#include "Anull.hpp"
#include "Apsnr.hpp"
#include "Areverse.hpp"
#include "Asdr.hpp"
#include "Ashowinfo.hpp"
#include "Asisdr.hpp"
#include "Earwax.hpp"
#include "Volumedetect.hpp"
#include "Anullsink.hpp"
#include "Addroi.hpp"
#include "Alphaextract.hpp"
#include "Alphamerge.hpp"
#include "Amplify.hpp"
#include "Atadenoise.hpp"
#include "Avgblur.hpp"
#include "Backgroundkey.hpp"
#include "Bbox.hpp"
#include "Bench.hpp"
#include "Bilateral.hpp"
#include "Bitplanenoise.hpp"
#include "Blackdetect.hpp"
#include "Blend.hpp"
#include "Blockdetect.hpp"
#include "Blurdetect.hpp"
#include "Bm3d.hpp"
#include "Bwdif.hpp"
#include "Cas.hpp"
#include "Ccrepack.hpp"
#include "Chromahold.hpp"
#include "Chromakey.hpp"
#include "Chromanr.hpp"
#include "Chromashift.hpp"
#include "Ciescope.hpp"
#include "Codecview.hpp"
#include "Colorbalance.hpp"
#include "Colorchannelmixer.hpp"
#include "Colorcontrast.hpp"
#include "Colorcorrect.hpp"
#include "Colorize.hpp"
#include "Colorkey.hpp"
#include "Colorhold.hpp"
#include "Colorlevels.hpp"
#include "Colormap.hpp"
#include "Colorspace.hpp"
#include "Colortemperature.hpp"
#include "Convolution.hpp"
#include "Convolve.hpp"
#include "Copy.hpp"
#include "Corr.hpp"
#include "Crop.hpp"
#include "Curves.hpp"
#include "Datascope.hpp"
#include "Dblur.hpp"
#include "Dctdnoiz.hpp"
#include "Deband.hpp"
#include "Deblock.hpp"
#include "Decimate.hpp"
#include "Deconvolve.hpp"
#include "Dedot.hpp"
#include "Deflate.hpp"
#include "Deflicker.hpp"
#include "Dejudder.hpp"
#include "Derain.hpp"
#include "Deshake.hpp"
#include "Despill.hpp"
#include "Detelecine.hpp"
#include "Dilation.hpp"
#include "Displace.hpp"
#include "Dnn_classify.hpp"
#include "Dnn_detect.hpp"
#include "Dnn_processing.hpp"
#include "Doubleweave.hpp"
#include "Drawbox.hpp"
#include "Drawgraph.hpp"
#include "Drawgrid.hpp"
#include "Edgedetect.hpp"
#include "Elbg.hpp"
#include "Entropy.hpp"
#include "Epx.hpp"
#include "Erosion.hpp"
#include "Estdif.hpp"
#include "Exposure.hpp"
#include "Extractplanes.hpp"
#include "Fade.hpp"
#include "Feedback.hpp"
#include "Fftdnoiz.hpp"
#include "Fftfilt.hpp"
#include "Field.hpp"
#include "Fieldhint.hpp"
#include "Fieldmatch.hpp"
#include "Fieldorder.hpp"
#include "Fillborders.hpp"
#include "Floodfill.hpp"
#include "Format.hpp"
#include "Fps.hpp"
#include "Framepack.hpp"
#include "Framerate.hpp"
#include "Framestep.hpp"
#include "Freezedetect.hpp"
#include "Freezeframes.hpp"
#include "Fsync.hpp"
#include "Gblur.hpp"
#include "Geq.hpp"
#include "Gradfun.hpp"
#include "Graphmonitor.hpp"
#include "Grayworld.hpp"
#include "Greyedge.hpp"
#include "Guided.hpp"
#include "Haldclut.hpp"
#include "Hflip.hpp"
#include "Histogram.hpp"
#include "Hqx.hpp"
#include "Hstack.hpp"
#include "Hsvhold.hpp"
#include "Hsvkey.hpp"
#include "Hue.hpp"
#include "Huesaturation.hpp"
#include "Hwdownload.hpp"
#include "Hwmap.hpp"
#include "Hwupload.hpp"
#include "Hwupload_cuda.hpp"
#include "Hysteresis.hpp"
#include "Identity.hpp"
#include "Idet.hpp"
#include "Il.hpp"
#include "Inflate.hpp"
#include "Interleave.hpp"
#include "Kirsch.hpp"
#include "Lagfun.hpp"
#include "Latency.hpp"
#include "Lenscorrection.hpp"
#include "Limitdiff.hpp"
#include "Limiter.hpp"
#include "Loop.hpp"
#include "Lumakey.hpp"
#include "Lut.hpp"
#include "Lut1d.hpp"
#include "Lut2.hpp"
#include "Lut3d.hpp"
#include "Lutrgb.hpp"
#include "Lutyuv.hpp"
#include "Maskedclamp.hpp"
#include "Maskedmax.hpp"
#include "Maskedmerge.hpp"
#include "Maskedmin.hpp"
#include "Maskedthreshold.hpp"
#include "Maskfun.hpp"
#include "Median.hpp"
#include "Mergeplanes.hpp"
#include "Mestimate.hpp"
#include "Metadata.hpp"
#include "Midequalizer.hpp"
#include "Minterpolate.hpp"
#include "Mix.hpp"
#include "Monochrome.hpp"
#include "Morpho.hpp"
#include "Msad.hpp"
#include "Multiply.hpp"
#include "Negate.hpp"
#include "Nlmeans.hpp"
#include "Noformat.hpp"
#include "Noise.hpp"
#include "Normalize.hpp"
#include "Null.hpp"
#include "Oscilloscope.hpp"
#include "Overlay.hpp"
#include "Pad.hpp"
#include "Palettegen.hpp"
#include "Paletteuse.hpp"
#include "Photosensitivity.hpp"
#include "Pixdesctest.hpp"
#include "Pixelize.hpp"
#include "Pixscope.hpp"
#include "Premultiply.hpp"
#include "Prewitt.hpp"
#include "Pseudocolor.hpp"
#include "Psnr.hpp"
#include "Qp.hpp"
#include "Random.hpp"
#include "Readeia608.hpp"
#include "Readvitc.hpp"
#include "Remap.hpp"
#include "Removegrain.hpp"
#include "Removelogo.hpp"
#include "Reverse.hpp"
#include "Rgbashift.hpp"
#include "Roberts.hpp"
#include "Rotate.hpp"
#include "Scale.hpp"
#include "Scale2ref.hpp"
#include "Scdet.hpp"
#include "Scharr.hpp"
#include "Scroll.hpp"
#include "Segment.hpp"
#include "Select.hpp"
#include "Selectivecolor.hpp"
#include "Separatefields.hpp"
#include "Setdar.hpp"
#include "Setfield.hpp"
#include "Setparams.hpp"
#include "Setpts.hpp"
#include "Setrange.hpp"
#include "Setsar.hpp"
#include "Settb.hpp"
#include "Shear.hpp"
#include "Showinfo.hpp"
#include "Showpalette.hpp"
#include "Shuffleframes.hpp"
#include "Shufflepixels.hpp"
#include "Shuffleplanes.hpp"
#include "Sidedata.hpp"
#include "Signalstats.hpp"
#include "Siti.hpp"
#include "Sobel.hpp"
#include "Sr.hpp"
#include "Ssim.hpp"
#include "Ssim360.hpp"
#include "Swaprect.hpp"
#include "Swapuv.hpp"
#include "Tblend.hpp"
#include "Telecine.hpp"
#include "Thistogram.hpp"
#include "Threshold.hpp"
#include "Thumbnail.hpp"
#include "Tile.hpp"
#include "Tiltandshift.hpp"
#include "Tlut2.hpp"
#include "Tmedian.hpp"
#include "Tmidequalizer.hpp"
#include "Tmix.hpp"
#include "Tonemap.hpp"
#include "Tpad.hpp"
#include "Transpose.hpp"
#include "Trim.hpp"
#include "Unpremultiply.hpp"
#include "Unsharp.hpp"
#include "Untile.hpp"
#include "V360.hpp"
#include "Varblur.hpp"
#include "Vectorscope.hpp"
#include "Vflip.hpp"
#include "Vfrdet.hpp"
#include "Vibrance.hpp"
#include "Vif.hpp"
#include "Vignette.hpp"
#include "Vmafmotion.hpp"
#include "Vstack.hpp"
#include "W3fdif.hpp"
#include "Waveform.hpp"
#include "Weave.hpp"
#include "Xbr.hpp"
#include "Xcorrelate.hpp"
#include "Xfade.hpp"
#include "Xmedian.hpp"
#include "Xstack.hpp"
#include "Yadif.hpp"
#include "Yaepblur.hpp"
#include "Zoompan.hpp"
#include "Allrgb.hpp"
#include "Allyuv.hpp"
#include "Cellauto.hpp"
#include "Color.hpp"
#include "Colorchart.hpp"
#include "Colorspectrum.hpp"
#include "Ddagrab.hpp"
#include "Gradients.hpp"
#include "Haldclutsrc.hpp"
#include "Life.hpp"
#include "Mandelbrot.hpp"
#include "Nullsrc.hpp"
#include "Pal75bars.hpp"
#include "Pal100bars.hpp"
#include "Rgbtestsrc.hpp"
#include "Sierpinski.hpp"
#include "Smptebars.hpp"
#include "Smptehdbars.hpp"
#include "Testsrc.hpp"
#include "Testsrc2.hpp"
#include "Yuvtestsrc.hpp"
#include "Zoneplate.hpp"
#include "Nullsink.hpp"
#include "A3dscope.hpp"
#include "Abitscope.hpp"
#include "Adrawgraph.hpp"
#include "Agraphmonitor.hpp"
#include "Ahistogram.hpp"
#include "Aphasemeter.hpp"
#include "Avectorscope.hpp"
#include "Showcqt.hpp"
#include "Showcwt.hpp"
#include "Showfreqs.hpp"
#include "Showspatial.hpp"
#include "Showspectrum.hpp"
#include "Showspectrumpic.hpp"
#include "Showvolume.hpp"
#include "Showwaves.hpp"
#include "Showwavespic.hpp"
#include "Buffer.hpp"
#include "Buffersink.hpp"
