import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, ImageOff, X, ZoomIn } from 'lucide-react'
import { useAuthedImage } from '../../hooks/useAuthedImage'
import { cn } from '../../lib/cn'

/**
 * An <img> for a protected media endpoint.
 * The bytes are fetched with the auth header and handed over as a blob: URL.
 */
export function AuthedImage({ path, alt, className, onClick, fit = 'cover' }) {
  const { url, loading, error } = useAuthedImage(path)

  if (loading) {
    return (
      <div className={cn('animate-pulse-soft bg-surface-3', className)} aria-label="Loading image" />
    )
  }

  if (error || !url) {
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center gap-1 bg-surface-2 text-ink-faint',
          className
        )}
      >
        <ImageOff className="h-4 w-4" />
        <span className="px-2 text-center text-[10px]">Unavailable</span>
      </div>
    )
  }

  return (
    <img
      src={url}
      alt={alt || ''}
      loading="lazy"
      onClick={onClick}
      className={cn(fit === 'cover' ? 'object-cover' : 'object-contain', className)}
    />
  )
}

/** Full-screen viewer with keyboard navigation. */
export function Lightbox({ images, index, onClose, onNavigate }) {
  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') onClose()
      if (event.key === 'ArrowRight') onNavigate(Math.min(images.length - 1, index + 1))
      if (event.key === 'ArrowLeft') onNavigate(Math.max(0, index - 1))
    }
    document.addEventListener('keydown', onKey)
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = previous
    }
  }, [index, images.length, onClose, onNavigate])

  const image = images[index]
  if (!image) return null

  return (
    <div className="fixed inset-0 z-[70] flex flex-col bg-black/90 animate-fade-in">
      <header className="flex items-start justify-between gap-4 px-5 py-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-ink">
            {image.caption || image.filename}
          </p>
          <p className="text-xs text-ink-faint">
            {image.page_no != null && `Page ${image.page_no} · `}
            {image.width} × {image.height}
            {images.length > 1 && ` · ${index + 1} of ${images.length}`}
          </p>
        </div>
        <button
          onClick={onClose}
          className="rounded-md p-1.5 text-ink-muted hover:bg-white/10 hover:text-ink"
          aria-label="Close"
        >
          <X className="h-5 w-5" />
        </button>
      </header>

      <div className="relative flex flex-1 items-center justify-center overflow-hidden px-4 pb-6">
        {index > 0 && (
          <button
            onClick={() => onNavigate(index - 1)}
            className="absolute left-4 rounded-full bg-white/10 p-2 text-ink hover:bg-white/20"
            aria-label="Previous image"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
        )}

        <AuthedImage
          path={`/media/${image.id}`}
          alt={image.caption || image.filename}
          fit="contain"
          className="max-h-full max-w-full rounded-lg"
        />

        {index < images.length - 1 && (
          <button
            onClick={() => onNavigate(index + 1)}
            className="absolute right-4 rounded-full bg-white/10 p-2 text-ink hover:bg-white/20"
            aria-label="Next image"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        )}
      </div>
    </div>
  )
}

/** The strip of figures rendered beneath an answer. */
export function ImageGallery({ images, compact = false }) {
  const [lightboxIndex, setLightboxIndex] = useState(null)

  if (!images?.length) return null

  return (
    <>
      <div className="mt-3 space-y-1.5">
        {!compact && (
          <span className="text-[10px] font-medium uppercase tracking-wide text-ink-faint">
            Related figures
          </span>
        )}
        <div
          className={cn(
            'grid gap-2',
            images.length === 1 ? 'grid-cols-1 max-w-sm' : 'grid-cols-2 sm:grid-cols-3'
          )}
        >
          {images.map((image, position) => (
            <figure
              key={image.id}
              className="group relative overflow-hidden rounded-lg border border-line bg-surface-2"
            >
              <button
                onClick={() => setLightboxIndex(position)}
                className="block w-full"
                aria-label="Open image"
              >
                <AuthedImage
                  path={image.thumb_url ? `/media/${image.id}/thumb` : `/media/${image.id}`}
                  alt={image.caption || image.filename}
                  className={cn(
                    'w-full transition-transform duration-200 group-hover:scale-[1.03]',
                    images.length === 1 ? 'max-h-64' : 'h-28'
                  )}
                />
                <span className="absolute right-1.5 top-1.5 rounded-md bg-black/55 p-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <ZoomIn className="h-3 w-3 text-white" />
                </span>
              </button>
              {image.caption && (
                <figcaption className="line-clamp-2 border-t border-line px-2 py-1.5 text-[10px] leading-snug text-ink-faint">
                  {image.caption}
                </figcaption>
              )}
            </figure>
          ))}
        </div>
      </div>

      {lightboxIndex != null && (
        <Lightbox
          images={images}
          index={lightboxIndex}
          onClose={() => setLightboxIndex(null)}
          onNavigate={setLightboxIndex}
        />
      )}
    </>
  )
}
