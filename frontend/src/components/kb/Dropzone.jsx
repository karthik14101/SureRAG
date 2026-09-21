import { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  AlertTriangle,
  CheckCircle2,
  FileArchive,
  FileJson,
  FileText,
  FileType2,
  Image as ImageIcon,
  Loader2,
  Upload,
  X,
} from 'lucide-react'
import { kbApi } from '../../api/kb'
import { cn } from '../../lib/cn'
import { fileExtension, formatBytes } from '../../lib/format'
import { useToast } from '../../hooks/useToast'
import { Button, ProgressBar } from '../ui'

const ACCEPT = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
  'application/json': ['.json'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/webp': ['.webp'],
  'image/bmp': ['.bmp'],
  'image/gif': ['.gif'],
  'application/zip': ['.zip'],
  'application/x-zip-compressed': ['.zip'],
}

const ICONS = {
  pdf: FileText,
  docx: FileType2,
  txt: FileText,
  md: FileText,
  json: FileJson,
  zip: FileArchive,
  png: ImageIcon,
  jpg: ImageIcon,
  jpeg: ImageIcon,
  webp: ImageIcon,
  bmp: ImageIcon,
  gif: ImageIcon,
}

export function FileIcon({ name, className }) {
  const Icon = ICONS[fileExtension(name)] || FileText
  return <Icon className={className} />
}

export function Dropzone({ kbId, onUploaded, disabled }) {
  const [limits, setLimits] = useState({ max_upload_mb: 100, max_files_per_request: 50 })
  const [uploading, setUploading] = useState(false)
  const [queue, setQueue] = useState([])
  const toast = useToast()

  useEffect(() => {
    kbApi.limits().then(setLimits).catch(() => {})
  }, [])

  const onDrop = useCallback(
    async (accepted, rejected) => {
      if (rejected?.length) {
        for (const item of rejected.slice(0, 3)) {
          const reason = item.errors?.[0]?.code === 'file-too-large'
            ? `is larger than ${limits.max_upload_mb} MB`
            : 'is not a supported file type'
          toast.error(`"${item.file.name}" ${reason}.`)
        }
      }
      if (!accepted?.length) return

      if (accepted.length > limits.max_files_per_request) {
        toast.error(
          `At most ${limits.max_files_per_request} files per upload. Try a ZIP archive.`
        )
        return
      }

      setQueue(accepted.map((file) => ({ name: file.name, size: file.size, state: 'uploading' })))
      setUploading(true)

      try {
        const result = await kbApi.upload(kbId, accepted)

        setQueue((prev) =>
          prev.map((item) => ({
            ...item,
            state: result.accepted.includes(item.name) ? 'queued' : 'skipped',
          }))
        )

        if (result.skipped?.length) {
          for (const skip of result.skipped.slice(0, 4)) {
            toast.info(`Skipped "${skip.name}": ${skip.reason}`)
          }
        }

        if (result.total_files > 0) {
          toast.success(
            `${result.total_files} file${result.total_files === 1 ? '' : 's'} queued for processing.`
          )
          onUploaded?.(result.job_id)
        } else if (!result.skipped?.length) {
          toast.error('Nothing was queued. Check the file types and try again.')
        }

        // Clear the transient queue display once the backend has taken over.
        setTimeout(() => setQueue([]), 2500)
      } catch (err) {
        setQueue((prev) => prev.map((item) => ({ ...item, state: 'failed' })))
        toast.error(err.message || 'The upload failed.')
      } finally {
        setUploading(false)
      }
    },
    [kbId, limits, onUploaded, toast]
  )

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: limits.max_upload_mb * 1024 * 1024,
    disabled: disabled || uploading,
    noClick: true,
    noKeyboard: true,
  })

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={cn(
          'rounded-xl2 border-2 border-dashed px-6 py-9 text-center transition-colors',
          isDragActive
            ? 'border-brand bg-brand/[0.07]'
            : 'border-line bg-surface-2/50 hover:border-line-soft',
          (disabled || uploading) && 'pointer-events-none opacity-60'
        )}
      >
        <input {...getInputProps()} />

        <div className="mx-auto mb-3 w-fit rounded-xl border border-line bg-surface-2 p-3">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-brand-soft" />
          ) : (
            <Upload className={cn('h-5 w-5', isDragActive ? 'text-brand-soft' : 'text-ink-faint')} />
          )}
        </div>

        <p className="text-sm font-medium text-ink">
          {isDragActive ? 'Drop to upload' : uploading ? 'Uploading...' : 'Drop files here'}
        </p>
        <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-ink-muted">
          PDF, Word, text, Markdown, JSON, images, or a ZIP archive.
          <br />
          Up to {limits.max_upload_mb} MB per file. Archives are unpacked automatically.
        </p>

        <Button variant="secondary" size="sm" className="mt-3.5" onClick={open} disabled={uploading}>
          Browse files
        </Button>
      </div>

      {queue.length > 0 && (
        <ul className="space-y-1.5">
          {queue.map((item) => (
            <li
              key={item.name}
              className="flex items-center gap-2.5 rounded-lg border border-line bg-surface-2 px-3 py-2"
            >
              <FileIcon name={item.name} className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
              <span className="min-w-0 flex-1 truncate text-xs text-ink">{item.name}</span>
              <span className="shrink-0 text-[10px] text-ink-faint">{formatBytes(item.size)}</span>
              {item.state === 'uploading' && (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-brand-soft" />
              )}
              {item.state === 'queued' && (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" />
              )}
              {item.state === 'skipped' && (
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-warn" />
              )}
              {item.state === 'failed' && <X className="h-3.5 w-3.5 shrink-0 text-danger" />}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** Live progress for the background ingestion jobs. */
export function UploadQueue({ jobs, onDismiss }) {
  if (!jobs?.length) return null

  return (
    <div className="space-y-2">
      {jobs.map((job) => {
        const done = ['completed', 'completed_with_errors', 'failed'].includes(job.state)
        const failed = job.state === 'failed'
        const problems = (job.errors || []).filter((entry) => entry.error)

        return (
          <div key={job.id} className="rounded-lg border border-line bg-surface-2 px-3.5 py-3">
            <div className="flex items-center gap-2">
              {!done && <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-soft" />}
              {done && !failed && !problems.length && (
                <CheckCircle2 className="h-3.5 w-3.5 text-success" />
              )}
              {(failed || problems.length > 0) && done && (
                <AlertTriangle className="h-3.5 w-3.5 text-warn" />
              )}

              <span className="flex-1 text-xs font-medium text-ink">
                {done
                  ? failed
                    ? 'Processing failed'
                    : `Processed ${job.processed_files - job.failed_files} of ${job.total_files} file${job.total_files === 1 ? '' : 's'}`
                  : `Processing ${job.processed_files + 1} of ${job.total_files}...`}
              </span>

              {done && onDismiss && (
                <button
                  onClick={() => onDismiss(job.id)}
                  className="text-ink-faint hover:text-ink"
                  aria-label="Dismiss"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>

            {!done && (
              <>
                <ProgressBar value={job.progress} className="mt-2" />
                {job.current_file && (
                  <p className="mt-1.5 truncate text-[10px] text-ink-faint">
                    {job.stage ? `${job.stage}: ` : ''}
                    {job.current_file}
                  </p>
                )}
              </>
            )}

            {done && problems.length > 0 && (
              <ul className="mt-2 space-y-1 border-t border-line pt-2">
                {problems.slice(0, 5).map((entry, index) => (
                  <li key={index} className="text-[10px] leading-relaxed text-warn">
                    <span className="font-medium">{entry.file}</span>: {entry.error}
                  </li>
                ))}
                {problems.length > 5 && (
                  <li className="text-[10px] text-ink-faint">
                    and {problems.length - 5} more
                  </li>
                )}
              </ul>
            )}
          </div>
        )
      })}
    </div>
  )
}
