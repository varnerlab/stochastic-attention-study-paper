# save_pub.jl
# -----------------------------------------------------------------------------
# Publication figures are ALWAYS vector PDF, never PNG.
#
# `save_pub(path, img)` writes `<stem>.pdf` regardless of the extension given in
# `path` (so existing call sites that pass "Fig_foo.png" keep working and simply
# emit "Fig_foo.pdf"). The raster is rendered to a temporary PNG and wrapped
# losslessly into a PDF via ImageMagick with pixel interpolation disabled, so
# pixel grids stay crisp. No PNG is left behind.
#
# Requires ImageMagick on the PATH (`magick`). Scripts already `using Images`,
# which provides `save` for the temporary render.
# -----------------------------------------------------------------------------
using FileIO

function save_pub(path::AbstractString, img)
    pdf_path = first(splitext(path)) * ".pdf"
    tmp = tempname() * ".png"
    save(tmp, img)
    try
        run(`magick $tmp -define pdf:interpolate=false $pdf_path`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    return pdf_path
end
