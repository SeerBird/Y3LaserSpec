$excel = [System.Runtime.InteropServices.Marshal]::GetActiveObject("Excel.Application")
$wb = $excel.ActiveWorkbook
Write-Host "Workbook Name: " $wb.Name
$ws = $excel.ActiveSheet
Write-Host "Sheet Name: " $ws.Name
$used = $ws.UsedRange
$rows = $used.Rows.Count
$cols = $used.Columns.Count
for ($r = 1; $r -le $rows; $r++) {
    $line = @()
    for ($c = 1; $c -le $cols; $c++) {
        $val = $used.Cells.Item($r, $c).Text
        $line += $val
    }
    Write-Host ($line -join " , ")
}
