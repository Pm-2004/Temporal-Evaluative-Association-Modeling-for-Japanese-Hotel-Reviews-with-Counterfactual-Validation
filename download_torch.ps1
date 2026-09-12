$url = "https://download-r2.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
$file = "torch-2.5.1+cu121-cp311-cp311-win_amd64.whl"
$maxRetries = 50
$retryCount = 0
$success = $false

while (-not $success -and $retryCount -lt $maxRetries) {
    Write-Host "Downloading PyTorch... (Attempt $($retryCount + 1) of $maxRetries)"
    # Using native Windows curl.exe to download with resume support (-C -)
    curl.exe -C - -L -O $url
    
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 33) {
        # curl exit code 33 means file is already completely downloaded
        $success = $true
        Write-Host "Download complete!"
    } else {
        $retryCount++
        Write-Host "Download interrupted (Exit code $LASTEXITCODE). Resuming in 2 seconds..."
        Start-Sleep -Seconds 2
    }
}

if ($success) {
    Write-Host "Installing local wheel..."
    .venv\Scripts\pip.exe install $file
} else {
    Write-Host "Failed to download after $maxRetries attempts."
}
