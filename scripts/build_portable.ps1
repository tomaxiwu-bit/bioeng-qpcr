[CmdletBinding()]
param(
    [string]$SourceDirectory = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourceDirectory) {
    $SourceDirectory = Join-Path $projectRoot "dist\bioeng-qpcr"
}
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot "dist-release"
}

$source = Get-Item -LiteralPath $SourceDirectory
$executable = Join-Path $source.FullName "bioeng-qpcr.exe"
$internal = Join-Path $source.FullName "_internal"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Packaged executable not found: $executable"
}
if (-not (Test-Path -LiteralPath $internal -PathType Container)) {
    throw "Packaged _internal directory not found: $internal"
}

$versionLine = Select-String -LiteralPath (Join-Path $projectRoot "pyproject.toml") -Pattern '^version = "([^"]+)"$' | Select-Object -First 1
if (-not $versionLine) {
    throw "Could not determine the project version from pyproject.toml."
}
$version = $versionLine.Matches[0].Groups[1].Value
$releaseName = "bioeng-qpcr-$version-win64-portable"

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$output = (Get-Item -LiteralPath $OutputDirectory).FullName
$stage = Join-Path $output $releaseName
$archive = Join-Path $output "$releaseName.zip"
$archiveChecksum = "$archive.sha256"
foreach ($target in ($stage, $archive, $archiveChecksum)) {
    if (Test-Path -LiteralPath $target) {
        throw "Release target already exists; move or remove it before rebuilding: $target"
    }
}

$smoke = Start-Process -FilePath $executable -ArgumentList "--smoke-test" -PassThru -WindowStyle Hidden
if (-not $smoke.WaitForExit(60000)) {
    $smoke.Kill()
    throw "Packaged scientific smoke test exceeded 60 seconds."
}
if ($smoke.ExitCode -ne 0) {
    throw "Packaged scientific smoke test failed with exit code $($smoke.ExitCode)."
}

Copy-Item -LiteralPath $source.FullName -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "packaging\README-PORTABLE.txt") -Destination (Join-Path $stage "README-PORTABLE.txt")
Copy-Item -LiteralPath (Join-Path $projectRoot "LICENSE") -Destination (Join-Path $stage "LICENSE")
Copy-Item -LiteralPath (Join-Path $projectRoot "THIRD_PARTY_NOTICES.md") -Destination (Join-Path $stage "THIRD_PARTY_NOTICES.md")
Copy-Item -LiteralPath (Join-Path $internal "BUILD_INFO.json") -Destination (Join-Path $stage "BUILD_INFO.json")
$buildInfo = Get-Content -LiteralPath (Join-Path $stage "BUILD_INFO.json") -Raw | ConvertFrom-Json
if ($buildInfo.software_version -ne $version) {
    throw "The executable build does not match the release version."
}
Copy-Item -LiteralPath (Join-Path $projectRoot "docs") -Destination (Join-Path $stage "docs") -Recurse

$manifestPath = Join-Path $stage "MANIFEST.sha256"
$manifestLines = Get-ChildItem -LiteralPath $stage -File -Recurse |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($stage, $_.FullName).Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash *$relative"
    }
[System.IO.File]::WriteAllLines($manifestPath, $manifestLines, [System.Text.UTF8Encoding]::new($false))

Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $archiveChecksum,
    "$zipHash *$releaseName.zip`n",
    [System.Text.UTF8Encoding]::new($false)
)

$signature = Get-AuthenticodeSignature -LiteralPath $executable
$archiveFile = Get-Item -LiteralPath $archive
[pscustomobject]@{
    Version = $version
    SmokeTestExitCode = $smoke.ExitCode
    Archive = $archiveFile.FullName
    ArchiveSizeMB = [math]::Round($archiveFile.Length / 1MB, 2)
    ArchiveSHA256 = $zipHash.ToUpperInvariant()
    ExecutableSignature = $signature.Status
}

