# 设置路径
$exePath = "C:\Users\uncle\_git_clones\CARLA_Latest\WindowsNoEditor\CarlaUE4.exe"
$pythonScript = "C:\Users\uncle\_git_clones\CARLA_Latest\WindowsNoEditor\myDemo\auto_drive_with_sensors.py"

# 每组任务配置：Town名 和 要跑几次
$taskList = @(
    @{ Town = "Town03_2"; Repetitions = 5 },
    @{ Town = "Town03_3"; Repetitions = 5 },
    @{ Town = "Town03_4"; Repetitions = 5 }
)

$totalTaskCount = 0
foreach ($task in $taskList) { $totalTaskCount += $task.Repetitions }
$currentTask = 0

foreach ($task in $taskList) {
    $town = $task.Town
    $reps = $task.Repetitions

    for ($i = 0; $i -lt $reps; $i++) {
        $currentTask++

        Write-Host ""
        Write-Host "=============================" -ForegroundColor Yellow
        Write-Host "开始执行第 $currentTask / $totalTaskCount 组" -ForegroundColor Cyan
        Write-Host "地图：$town  | 第 $($i+1) 次" -ForegroundColor Green
        Write-Host "=============================" -ForegroundColor Yellow
        Write-Host ""

        # 1. 启动exe
        $exeProcess = Start-Process -FilePath $exePath -PassThru
        Start-Sleep -Seconds 5

        # 2. 运行Python脚本（不带repetitions，只传town和随机种子）
        $randomSeed = $i + 1  # 可以自定义种子生成逻辑
        python $pythonScript --town $town --random_seed $randomSeed

        # 3. 关闭exe（名字中包含CarlaUE4的全部杀掉）
        Get-Process | Where-Object { $_.Name -like "*CarlaUE4*" } | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host "所有任务执行完毕！" -ForegroundColor Magenta