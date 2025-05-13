# ========== 配置参数 ==========
$baseDir = "E:\datasets\mydataset\one_frame\2025_04_26_20_05_56"
$agents = 1..4  # 无人机编号
$baseFileName = "283"
$copies = 500    # 复制几份
$step = 10      # 每次递增编号
$extensions = @(".json", ".bin", ".pcd", ".png", ".yaml")  # 支持的文件类型

# ========== 主逻辑 ==========
foreach ($agent in $agents) {
    $agentDir = Join-Path $baseDir "$agent"

    for ($i = 1; $i -le $copies; $i++) {
        $newIndex = [int]$baseFileName + ($i * $step)
        $newName = "{0:D6}" -f $newIndex

        foreach ($ext in $extensions) {
            $srcFile = Join-Path $agentDir ("$baseFileName$ext")
            $dstFile = Join-Path $agentDir ("$newName$ext")

            if (Test-Path $srcFile) {
                Copy-Item $srcFile $dstFile -Force
                Write-Output "Agent ${agent}: 复制 ${srcFile} → ${dstFile}"
            } else {
                Write-Warning "Agent ${agent}: 源文件不存在 ${srcFile}，已跳过"
            }
        }
    }
}
