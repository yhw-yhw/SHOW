### Origin Youtube Video Links

<!-- * Standup Comedy: 
 > https://www.youtube.com/channel/UCxyCzPY2pjAjrxoSYclpuLg
 > number: 2196
* TED Talk: 
 > https://www.youtube.com/channel/UCAuUUnT6oDeKwE6v1NGQxug
 > number: 4000 -->

[@metoffice](https://www.youtube.com/@metoffice)
https://www.youtube.com/channel/UC40Tw2tFuMzK305mi7nj8rg
numbers: 8000

BBC news:
https://www.youtube.com/channel/UC16niRr50-MSBwiO3YDb3RA

Standup Comedy: 
https://www.youtube.com/channel/UCxyCzPY2pjAjrxoSYclpuLg
number: 2196

TED Talk: 
https://www.youtube.com/channel/UCAuUUnT6oDeKwE6v1NGQxug
number: 4000

CNN news:
https://www.youtube.com/playlist?list=PL6XRrncXkMaW5p7muaR2s2IqjouQh4jqS

World news from the BBC:
https://www.youtube.com/playlist?list=PLS3XGZxi7cBVTzEE4Sim9UuNKnUJq9Vkh

News | National Geographic:
https://www.youtube.com/playlist?list=PLivjPDlt6ApRfQqtRw7JkGCLvezGeMBB2

TVB Weather Report:
https://www.youtube.com/playlist?list=PLtQjrt9Q28IX5HLZfWGyI3gxRmJpnDF_l
numbers: 813

Idaho Weather:
https://www.youtube.com/playlist?list=PLggbABFJJpUALac2V9Epvz1E95aoesh4y
numbers: 1416

民視氣象預報 FTV Weather Forecast:
https://www.youtube.com/playlist?list=PL05E88006D42B812E
numbers: 947

新聞Talk Show:
https://www.youtube.com/playlist?list=PLSg6_lakxpXEfa-H7CUq-msrBZQCRGrub
numbers: 1135

TV Talk Shows & Late Night:
https://www.youtube.com/playlist?list=PLrEnWoR732-CN09YykVof2lxdI3MLOZda
numbers: 2969


Youtube search keywords:
news, weather forecast, talk show, 


```shell
yt-dlp --format "bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" https://www.youtube.com/watch?v=F9XB29JfKYo

yt-dlp --format "bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" https://www.youtube.com/channel/UCxyCzPY2pjAjrxoSYclpuLg

yt-dlp --format "bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" https://www.youtube.com/channel/UCAuUUnT6oDeKwE6v1NGQxug

yt-dlp --format "bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" https://www.youtube.com/channel/UC16niRr50-MSBwiO3YDb3RA
```

find youtube channel ID:
1\view shource; find www.youtube.com/channel/
2\Request Payload under the browseId
3\https://www.googleapis.com/youtube/v3/videos?part=snippet&id=tlNBB0jdL3w
4\https://www.streamweasels.com/tools/youtube-channel-id-and-user-id-convertor/


yt-dlp --continue -o "%(playlist_index)s-%(title)s.%(ext)s" https://www.youtube.com/channel/UC40Tw2tFuMzK305mi7nj8rg
