"""
天气查询工具
提供天气信息查询功能（模拟和真实API）
"""

import os
import requests
from typing import Dict, Optional
from datetime import datetime


class WeatherTool:
    """天气查询工具类"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化天气工具

        Args:
            api_key: OpenWeatherMap API密钥（可选）
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

        # 模拟数据（当没有API密钥时使用）
        self.mock_data = {
            "北京": {"temp": 15, "condition": "晴", "humidity": 45, "wind": 12},
            "上海": {"temp": 20, "condition": "多云", "humidity": 60, "wind": 15},
            "深圳": {"temp": 25, "condition": "小雨", "humidity": 75, "wind": 8},
            "广州": {"temp": 24, "condition": "阴", "humidity": 70, "wind": 10},
            "成都": {"temp": 18, "condition": "阴", "humidity": 70, "wind": 5},
            "杭州": {"temp": 19, "condition": "多云", "humidity": 65, "wind": 13},
            "南京": {"temp": 17, "condition": "晴", "humidity": 50, "wind": 11},
            "武汉": {"temp": 16, "condition": "多云", "humidity": 55, "wind": 9},
        }

    def get_weather(self, city: str) -> str:
        """
        获取城市天气信息

        Args:
            city: 城市名称

        Returns:
            格式化的天气信息字符串
        """
        # 如果有API密钥，尝试使用真实API
        if self.api_key:
            try:
                return self._get_real_weather(city)
            except Exception as e:
                print(f"API调用失败，使用模拟数据: {e}")
                return self._get_mock_weather(city)
        else:
            return self._get_mock_weather(city)

    def _get_real_weather(self, city: str) -> str:
        """
        从OpenWeatherMap API获取真实天气数据

        Args:
            city: 城市名称

        Returns:
            格式化的天气信息
        """
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',  # 使用摄氏度
            'lang': 'zh_cn'
        }

        response = requests.get(self.base_url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()

        # 解析数据
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        wind_speed = data['wind']['speed']

        # 格式化输出
        weather_info = f"""🌤️ {city}天气信息:
━━━━━━━━━━━━━━━━━━━━
🌡️  温度: {temp}°C (体感 {feels_like}°C)
☁️  天气: {description}
💧 湿度: {humidity}%
💨 风速: {wind_speed} m/s
⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━
"""
        return weather_info

    def _get_mock_weather(self, city: str) -> str:
        """
        获取模拟天气数据

        Args:
            city: 城市名称

        Returns:
            格式化的天气信息
        """
        if city in self.mock_data:
            data = self.mock_data[city]
            weather_info = f"""🌤️ {city}天气信息 (模拟数据):
━━━━━━━━━━━━━━━━━━━━
🌡️  温度: {data['temp']}°C
☁️  天气: {data['condition']}
💧 湿度: {data['humidity']}%
💨 风速: {data['wind']} km/h
⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━
💡 提示: 这是模拟数据，配置OPENWEATHER_API_KEY可获取真实数据
"""
            return weather_info
        else:
            supported_cities = ", ".join(self.mock_data.keys())
            return f"❌ 暂无{city}的天气数据\n\n支持的城市: {supported_cities}"

    def get_description(self) -> str:
        """获取工具描述"""
        return """天气查询工具 - 获取城市天气信息

功能:
  • 查询城市当前天气
  • 温度、湿度、风速等信息
  • 支持真实API和模拟数据

使用方法:
  输入城市名称，例如: "北京", "上海", "深圳"

支持的城市（模拟数据）:
  北京、上海、深圳、广州、成都、杭州、南京、武汉

配置真实API:
  1. 注册OpenWeatherMap账号: https://openweathermap.org/
  2. 获取API密钥
  3. 设置环境变量: OPENWEATHER_API_KEY=your_key
"""


# 便捷函数
def get_weather(city: str, api_key: Optional[str] = None) -> str:
    """
    便捷的天气查询函数

    Args:
        city: 城市名称
        api_key: API密钥（可选）

    Returns:
        天气信息
    """
    tool = WeatherTool(api_key)
    return tool.get_weather(city)


# 测试代码
if __name__ == "__main__":
    tool = WeatherTool()

    print("🌤️  天气查询工具测试\n")
    print(tool.get_description())

    print("\n" + "=" * 50)
    print("测试查询:")
    print("=" * 50)

    test_cities = ["北京", "上海", "深圳", "不存在的城市"]

    for city in test_cities:
        print(f"\n查询: {city}")
        print("-" * 50)
        result = tool.get_weather(city)
        print(result)
